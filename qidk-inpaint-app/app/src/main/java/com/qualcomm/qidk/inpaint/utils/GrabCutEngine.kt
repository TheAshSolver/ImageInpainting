package com.qualcomm.qidk.inpaint.utils

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import java.util.ArrayDeque
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

object GrabCutEngine {

    /**
     * SOTA Interactive Smart Object Extractor.
     * Takes an arbitrary user prompt mask (whether created by freehand brush or solid square box),
     * analyzes background context, estimates multi-modal 3D color distributions, evaluates
     * image contrast-sensitive edge potentials, and executes globally optimal Min-Cut/Max-Flow.
     * Refines the resulting silhouette with morphological hole filling and border snapping.
     */
    fun extractSmartObject(source: Bitmap, roughMask: Bitmap): Pair<Bitmap, Long> {
        val t0 = System.currentTimeMillis()
        val w = source.width
        val h = source.height
        val outBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        val srcPixels = IntArray(w * h)
        val maskPixels = IntArray(w * h)
        source.getPixels(srcPixels, 0, w, 0, 0, w, h)
        roughMask.getPixels(maskPixels, 0, w, 0, 0, w, h)

        val outPixels = extractSmartObjectFromPixels(w, h, srcPixels, maskPixels)
        outBmp.setPixels(outPixels, 0, w, 0, 0, w, h)

        val latency = System.currentTimeMillis() - t0
        return Pair(outBmp, latency)
    }

    /**
     * Legacy adapter for Bounding Box mask extraction.
     */
    fun generateBoundingBoxMask(source: Bitmap, bbox: RectF): Pair<Bitmap, Long> {
        val w = source.width
        val h = source.height
        val boxMask = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        val left = max(0, bbox.left.toInt())
        val top = max(0, bbox.top.toInt())
        val right = min(w - 1, bbox.right.toInt())
        val bottom = min(h - 1, bbox.bottom.toInt())

        for (y in top..bottom) {
            val row = y * w
            for (x in left..right) {
                pixels[row + x] = COLOR_WHITE
            }
        }
        boxMask.setPixels(pixels, 0, w, 0, 0, w, h)
        return extractSmartObject(source, boxMask)
    }

    /**
     * Legacy adapter for snap to edges.
     */
    fun refineMaskSnapToEdges(source: Bitmap, roughMask: Bitmap): Pair<Bitmap, Long> {
        return extractSmartObject(source, roughMask)
    }

    private fun colorA(c: Int): Int = (c ushr 24) and 0xFF
    private fun colorR(c: Int): Int = (c ushr 16) and 0xFF
    private fun colorG(c: Int): Int = (c ushr 8) and 0xFF
    private fun colorB(c: Int): Int = c and 0xFF
    private const val COLOR_WHITE = -0x1 // 0xFFFFFFFF

    private fun isHolePixel(c: Int): Boolean {
        val a = colorA(c)
        val r = colorR(c)
        val g = colorG(c)
        val b = colorB(c)
        val luma = (r * 299 + g * 587 + b * 114) / 1000
        return a > 50 && (luma > 128 || r > 128)
    }

    /**
     * Core pure-algorithm segmentation pipeline.
     */
    fun extractSmartObjectFromPixels(
        w: Int,
        h: Int,
        srcPixels: IntArray,
        maskPixels: IntArray
    ): IntArray {
        val outPixels = IntArray(w * h)

        // 1. Scan for active mask bounding box
        var minX = w
        var maxX = -1
        var minY = h
        var maxY = -1
        var maskCount = 0

        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                if (isHolePixel(maskPixels[row + x])) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                    maskCount++
                }
            }
        }

        if (maskCount == 0 || minX > maxX || minY > maxY) {
            return outPixels
        }

        val boxW = maxX - minX + 1
        val boxH = maxY - minY + 1

        // 2. Establish context margin around mask
        val margin = max(16, min(56, max(boxW, boxH) / 5))
        val roiX1 = max(0, minX - margin)
        val roiY1 = max(0, minY - margin)
        val roiX2 = min(w - 1, maxX + margin)
        val roiY2 = min(h - 1, maxY + margin)
        val roiW = roiX2 - roiX1 + 1
        val roiH = roiY2 - roiY1 + 1

        // 3. Grid sizing for Graph-Cut (Max 144px along longest dimension for sub-25ms inference)
        val maxDim = 144
        val scale = if (max(roiW, roiH) > maxDim) {
            maxDim.toFloat() / max(roiW, roiH)
        } else {
            1.0f
        }

        val gridW = max(16, (roiW * scale).roundToInt())
        val gridH = max(16, (roiH * scale).roundToInt())
        val N = gridW * gridH

        val gridColors = IntArray(N)
        val gridMask = BooleanArray(N)

        for (gy in 0 until gridH) {
            val srcY = min(roiY2, roiY1 + (gy.toFloat() / scale).roundToInt())
            val gRow = gy * gridW
            val sRow = srcY * w
            for (gx in 0 until gridW) {
                val srcX = min(roiX2, roiX1 + (gx.toFloat() / scale).roundToInt())
                val idx = gRow + gx
                gridColors[idx] = srcPixels[sRow + srcX]
                gridMask[idx] = isHolePixel(maskPixels[sRow + srcX])
            }
        }

        // 4. Two-Pass Manhattan Distance Transform on grid to identify deep core seeds
        val distToBg = FloatArray(N)
        val infDist = 10000.0f
        for (i in 0 until N) {
            distToBg[i] = if (gridMask[i]) infDist else 0.0f
        }

        // Forward pass
        for (gy in 0 until gridH) {
            val row = gy * gridW
            for (gx in 0 until gridW) {
                val idx = row + gx
                if (gridMask[idx]) {
                    var d = distToBg[idx]
                    if (gx > 0) d = min(d, distToBg[idx - 1] + 1f)
                    if (gy > 0) d = min(d, distToBg[idx - gridW] + 1f)
                    distToBg[idx] = d
                }
            }
        }

        // Backward pass
        var maxDist = 0.0f
        for (gy in gridH - 1 downTo 0) {
            val row = gy * gridW
            for (gx in gridW - 1 downTo 0) {
                val idx = row + gx
                if (gridMask[idx]) {
                    var d = distToBg[idx]
                    if (gx < gridW - 1) d = min(d, distToBg[idx + 1] + 1f)
                    if (gy < gridH - 1) d = min(d, distToBg[idx + gridW] + 1f)
                    distToBg[idx] = d
                    if (d > maxDist) maxDist = d
                }
            }
        }

        // Seed thresholds
        val seedThreshold = max(1.5f, maxDist * 0.45f)

        // 5. 3D Perceptual Color Histograms (16 x 16 x 16 = 4096 bins)
        val bgHist = IntArray(4096)
        val fgHist = IntArray(4096)
        var bgSamples = 0
        var fgSamples = 0

        fun colorToBin(c: Int): Int {
            val r = (colorR(c) ushr 4) and 0x0F
            val g = (colorG(c) ushr 4) and 0x0F
            val b = (colorB(c) ushr 4) and 0x0F
            return (r shl 8) or (g shl 4) or b
        }

        for (i in 0 until N) {
            val c = gridColors[i]
            val b = colorToBin(c)
            if (!gridMask[i]) {
                bgHist[b]++
                bgSamples++
            } else if (distToBg[i] >= seedThreshold) {
                fgHist[b]++
                fgSamples++
            }
        }

        // If foreground core was too tiny, sample center region of mask
        if (fgSamples < 5) {
            val fgTargetDist = max(0.5f, maxDist * 0.2f)
            for (i in 0 until N) {
                if (gridMask[i] && distToBg[i] >= fgTargetDist) {
                    val b = colorToBin(gridColors[i])
                    fgHist[b]++
                    fgSamples++
                }
            }
        }

        // 3D Kernel Smoothing across neighboring bins to eliminate zero-frequency issues
        val smoothBg = FloatArray(4096)
        val smoothFg = FloatArray(4096)
        var totalBgSmoothed = 0.0f
        var totalFgSmoothed = 0.0f

        for (r in 0 until 16) {
            for (g in 0 until 16) {
                for (b in 0 until 16) {
                    val idx = (r shl 8) or (g shl 4) or b
                    var bgSum = bgHist[idx] * 4.0f
                    var fgSum = fgHist[idx] * 4.0f

                    if (r > 0) { val n = ((r - 1) shl 8) or (g shl 4) or b; bgSum += bgHist[n]; fgSum += fgHist[n] }
                    if (r < 15) { val n = ((r + 1) shl 8) or (g shl 4) or b; bgSum += bgHist[n]; fgSum += fgHist[n] }
                    if (g > 0) { val n = (r shl 8) or ((g - 1) shl 4) or b; bgSum += bgHist[n]; fgSum += fgHist[n] }
                    if (g < 15) { val n = (r shl 8) or ((g + 1) shl 4) or b; bgSum += bgHist[n]; fgSum += fgHist[n] }
                    if (b > 0) { val n = (r shl 8) or (g shl 4) or (b - 1); bgSum += bgHist[n]; fgSum += fgHist[n] }
                    if (b < 15) { val n = (r shl 8) or (g shl 4) or (b + 1); bgSum += bgHist[n]; fgSum += fgHist[n] }

                    smoothBg[idx] = bgSum
                    smoothFg[idx] = fgSum
                    totalBgSmoothed += bgSum
                    totalFgSmoothed += fgSum
                }
            }
        }

        val denomBg = max(1.0f, totalBgSmoothed) + 4096.0f
        val denomFg = max(1.0f, totalFgSmoothed) + 4096.0f

        // 6. Contrast-Attenuated Edge Weights
        var sumDiffSq = 0.0
        var edgeCount = 0

        for (gy in 0 until gridH) {
            val row = gy * gridW
            for (gx in 0 until gridW) {
                val idx = row + gx
                val c1 = gridColors[idx]
                val r1 = colorR(c1); val g1 = colorG(c1); val b1 = colorB(c1)

                if (gx + 1 < gridW) {
                    val c2 = gridColors[idx + 1]
                    val dr = r1 - colorR(c2); val dg = g1 - colorG(c2); val db = b1 - colorB(c2)
                    sumDiffSq += (dr * dr + dg * dg + db * db)
                    edgeCount++
                }
                if (gy + 1 < gridH) {
                    val c2 = gridColors[idx + gridW]
                    val dr = r1 - colorR(c2); val dg = g1 - colorG(c2); val db = b1 - colorB(c2)
                    sumDiffSq += (dr * dr + dg * dg + db * db)
                    edgeCount++
                }
            }
        }

        val beta = if (edgeCount > 0) {
            1.0f / (2.0f * max(1.0f, (sumDiffSq / edgeCount).toFloat()))
        } else {
            0.005f
        }
        val gamma = 45.0f

        // 7. Graph-Cut Construction (Dinic Flow Network)
        val sourceNode = N
        val sinkNode = N + 1
        val totalNodes = N + 2
        val dinic = DinicSolver(totalNodes)

        val infCap = 10000.0f
        val centerX = gridW / 2.0f
        val centerY = gridH / 2.0f
        val maxRadSq = max(1.0f, (gridW * gridW + gridH * gridH) / 4.0f)

        for (gy in 0 until gridH) {
            val row = gy * gridW
            for (gx in 0 until gridW) {
                val idx = row + gx
                val isMask = gridMask[idx]
                val c = gridColors[idx]
                val b = colorToBin(c)

                val pBg = (smoothBg[b] + 1.0f) / denomBg
                val pFg = (smoothFg[b] + 1.0f) / denomFg

                val costBg = -ln(pBg)
                val costFg = -ln(pFg)

                val isDefiniteBg = !isMask && (gx <= 2 || gx >= gridW - 3 || gy <= 2 || gy >= gridH - 3)
                val isDefiniteFg = isMask && distToBg[idx] >= seedThreshold && seedThreshold >= 2.0f

                if (isDefiniteBg) {
                    dinic.addEdge(idx, sinkNode, infCap)
                } else if (isDefiniteFg) {
                    dinic.addEdge(sourceNode, idx, infCap)
                } else if (!isMask) {
                    // Pixels outside user mask strongly penalized if labeled foreground
                    dinic.addEdge(idx, sinkNode, costBg + 8.0f)
                } else {
                    // Inside user mask
                    val dx = gx - centerX
                    val dy = gy - centerY
                    val radSq = dx * dx + dy * dy
                    val spatialPrior = 2.5f * max(0.0f, 1.0f - (radSq / maxRadSq))

                    dinic.addEdge(sourceNode, idx, costBg + spatialPrior)
                    dinic.addEdge(idx, sinkNode, costFg)
                }

                // Neighbor edges
                val r1 = colorR(c); val g1 = colorG(c); val b1 = colorB(c)
                if (gx + 1 < gridW) {
                    val c2 = gridColors[idx + 1]
                    val dr = r1 - colorR(c2); val dg = g1 - colorG(c2); val db = b1 - colorB(c2)
                    val cap = gamma * exp(-beta * (dr * dr + dg * dg + db * db))
                    dinic.addUndirectedEdge(idx, idx + 1, cap)
                }
                if (gy + 1 < gridH) {
                    val c2 = gridColors[idx + gridW]
                    val dr = r1 - colorR(c2); val dg = g1 - colorG(c2); val db = b1 - colorB(c2)
                    val cap = gamma * exp(-beta * (dr * dr + dg * dg + db * db))
                    dinic.addUndirectedEdge(idx, idx + gridW, cap)
                }
            }
        }

        // Solve Min-Cut
        dinic.maxFlow(sourceNode, sinkNode)
        val inCut = dinic.getSourceComponent(sourceNode)

        // 8. Map Cut to ROI Boolean Grid
        val roiBinary = BooleanArray(roiW * roiH)

        if (scale >= 0.99f) {
            for (i in 0 until min(N, roiW * roiH)) {
                roiBinary[i] = inCut[i]
            }
        } else {
            // Bilateral-guided upsampling: snaps boundaries to high-res image gradients
            for (ry in 0 until roiH) {
                val gy = min(gridH - 1, (ry * scale).toInt())
                val rRow = ry * roiW
                val gRow = gy * gridW
                val srcRow = (roiY1 + ry) * w

                for (rx in 0 until roiW) {
                    val gx = min(gridW - 1, (rx * scale).toInt())
                    val gIdx = gRow + gx
                    val cutVal = inCut[gIdx]

                    // Check if near boundary
                    var isBoundary = false
                    if (gx > 0 && inCut[gIdx - 1] != cutVal) isBoundary = true
                    if (gx < gridW - 1 && inCut[gIdx + 1] != cutVal) isBoundary = true
                    if (gy > 0 && inCut[gIdx - gridW] != cutVal) isBoundary = true
                    if (gy < gridH - 1 && inCut[gIdx + gridW] != cutVal) isBoundary = true

                    if (!isBoundary) {
                        roiBinary[rRow + rx] = cutVal
                    } else {
                        // Bilateral snap using high-resolution local color
                        val cHigh = srcPixels[srcRow + (roiX1 + rx)]
                        val b = colorToBin(cHigh)
                        val pBg = (smoothBg[b] + 1.0f) / denomBg
                        val pFg = (smoothFg[b] + 1.0f) / denomFg
                        roiBinary[rRow + rx] = (pFg >= pBg)
                    }
                }
            }
        }

        // 9. Morphological Topological Hole Filling
        // Flood-fill background starting from outer boundary of ROI; any background not reached is an enclosed hole
        val bgVisited = BooleanArray(roiW * roiH)
        val qx = IntArray(roiW * roiH)
        val qy = IntArray(roiW * roiH)
        var qHead = 0
        var qTail = 0

        fun enqueue(x: Int, y: Int) {
            val idx = y * roiW + x
            if (!roiBinary[idx] && !bgVisited[idx]) {
                bgVisited[idx] = true
                qx[qTail] = x
                qy[qTail] = y
                qTail++
            }
        }

        for (x in 0 until roiW) {
            enqueue(x, 0)
            enqueue(x, roiH - 1)
        }
        for (y in 0 until roiH) {
            enqueue(0, y)
            enqueue(roiW - 1, y)
        }

        while (qHead < qTail) {
            val cx = qx[qHead]
            val cy = qy[qHead]
            qHead++

            if (cx > 0) enqueue(cx - 1, cy)
            if (cx < roiW - 1) enqueue(cx + 1, cy)
            if (cy > 0) enqueue(cx, cy - 1)
            if (cy < roiH - 1) enqueue(cx, cy + 1)
        }

        // Fill holes (unreached background inside object)
        for (i in 0 until (roiW * roiH)) {
            if (!roiBinary[i] && !bgVisited[i]) {
                roiBinary[i] = true
            }
        }

        // 10. Connected Component Island Filter (Discard tiny speckles < 2% of main object)
        val visited = BooleanArray(roiW * roiH)
        var maxCompSize = 0
        var mainCompId = -1
        val compSizes = mutableListOf<Int>()
        val compMap = IntArray(roiW * roiH) { -1 }

        for (ry in 0 until roiH) {
            val row = ry * roiW
            for (rx in 0 until roiW) {
                val idx = row + rx
                if (roiBinary[idx] && !visited[idx]) {
                    val compId = compSizes.size
                    var size = 0
                    qHead = 0
                    qTail = 0
                    visited[idx] = true
                    compMap[idx] = compId
                    qx[qTail] = rx; qy[qTail] = ry; qTail++

                    while (qHead < qTail) {
                        val cx = qx[qHead]
                        val cy = qy[qHead]
                        qHead++
                        size++

                        val neighbors = arrayOf(
                            Pair(cx - 1, cy), Pair(cx + 1, cy),
                            Pair(cx, cy - 1), Pair(cx, cy + 1)
                        )
                        for (p in neighbors) {
                            val nx = p.first; val ny = p.second
                            if (nx in 0 until roiW && ny in 0 until roiH) {
                                val nIdx = ny * roiW + nx
                                if (roiBinary[nIdx] && !visited[nIdx]) {
                                    visited[nIdx] = true
                                    compMap[nIdx] = compId
                                    qx[qTail] = nx; qy[qTail] = ny; qTail++
                                }
                            }
                        }
                    }

                    compSizes.add(size)
                    if (size > maxCompSize) {
                        maxCompSize = size
                        mainCompId = compId
                    }
                }
            }
        }

        val minValidCompSize = max(10, (maxCompSize * 0.04f).toInt())
        for (i in 0 until (roiW * roiH)) {
            val cId = compMap[i]
            if (cId == -1 || (cId != mainCompId && compSizes[cId] < minValidCompSize)) {
                roiBinary[i] = false
            }
        }

        // 11. Safety Margin Dilation (2 pixels) to guarantee full coverage without inpainting halos
        val dilated = BooleanArray(roiW * roiH)
        for (ry in 0 until roiH) {
            val row = ry * roiW
            for (rx in 0 until roiW) {
                if (roiBinary[row + rx]) {
                    for (dy in -2..2) {
                        val ny = ry + dy
                        if (ny in 0 until roiH) {
                            val nRow = ny * roiW
                            for (dx in -2..2) {
                                val nx = rx + dx
                                if (nx in 0 until roiW && (dx * dx + dy * dy <= 5)) {
                                    dilated[nRow + nx] = true
                                }
                            }
                        }
                    }
                }
            }
        }

        // 12. Write back to full-resolution output array
        for (ry in 0 until roiH) {
            val rRow = ry * roiW
            val outRow = (roiY1 + ry) * w
            for (rx in 0 until roiW) {
                if (dilated[rRow + rx]) {
                    outPixels[outRow + (roiX1 + rx)] = COLOR_WHITE
                }
            }
        }

        return outPixels
    }

    /**
     * Highly optimized Dinic's Algorithm for Min-Cut / Max-Flow.
     * Uses flat primitive arrays for zero object allocations and maximum L1 cache efficiency.
     */
    private class DinicSolver(private val n: Int) {
        private var edgeCap = 160000
        private var head = IntArray(n) { -1 }
        private var to = IntArray(edgeCap)
        private var cap = FloatArray(edgeCap)
        private var flow = FloatArray(edgeCap)
        private var nxt = IntArray(edgeCap)
        private var edgeCount = 0

        private var level = IntArray(n)
        private var ptr = IntArray(n)
        private var q = IntArray(n)

        private fun ensureCapacity() {
            if (edgeCount + 4 >= edgeCap) {
                edgeCap *= 2
                to = to.copyOf(edgeCap)
                cap = cap.copyOf(edgeCap)
                flow = flow.copyOf(edgeCap)
                nxt = nxt.copyOf(edgeCap)
            }
        }

        fun addEdge(from: Int, target: Int, capacity: Float) {
            ensureCapacity()
            to[edgeCount] = target; cap[edgeCount] = capacity; flow[edgeCount] = 0.0f; nxt[edgeCount] = head[from]; head[from] = edgeCount++
            to[edgeCount] = from; cap[edgeCount] = 0.0f; flow[edgeCount] = 0.0f; nxt[edgeCount] = head[target]; head[target] = edgeCount++
        }

        fun addUndirectedEdge(u: Int, v: Int, capacity: Float) {
            ensureCapacity()
            to[edgeCount] = v; cap[edgeCount] = capacity; flow[edgeCount] = 0.0f; nxt[edgeCount] = head[u]; head[u] = edgeCount++
            to[edgeCount] = u; cap[edgeCount] = capacity; flow[edgeCount] = 0.0f; nxt[edgeCount] = head[v]; head[v] = edgeCount++
        }

        private fun bfs(s: Int, t: Int): Boolean {
            level.fill(-1)
            level[s] = 0
            var qh = 0
            var qt = 0
            q[qt++] = s

            while (qh < qt) {
                val v = q[qh++]
                var e = head[v]
                while (e != -1) {
                    val trg = to[e]
                    if (level[trg] < 0 && cap[e] - flow[e] > 1e-4f) {
                        level[trg] = level[v] + 1
                        q[qt++] = trg
                    }
                    e = nxt[e]
                }
            }
            return level[t] >= 0
        }

        private fun dfs(v: Int, t: Int, pushed: Float): Float {
            if (pushed <= 1e-4f || v == t) return pushed

            var e = ptr[v]
            while (e != -1) {
                val trg = to[e]
                if (level[trg] == level[v] + 1 && cap[e] - flow[e] > 1e-4f) {
                    val tr = dfs(trg, t, min(pushed, cap[e] - flow[e]))
                    if (tr > 1e-4f) {
                        flow[e] += tr
                        flow[e xor 1] -= tr
                        return tr
                    }
                }
                e = nxt[e]
                ptr[v] = e
            }
            return 0.0f
        }

        fun maxFlow(s: Int, t: Int): Float {
            var totalFlow = 0.0f
            while (bfs(s, t)) {
                ptr.indices.forEach { ptr[it] = head[it] }
                while (true) {
                    val pushed = dfs(s, t, Float.MAX_VALUE)
                    if (pushed <= 1e-4f) break
                    totalFlow += pushed
                }
            }
            return totalFlow
        }

        fun getSourceComponent(s: Int): BooleanArray {
            val inCut = BooleanArray(n)
            val qVisited = IntArray(n)
            var qh = 0
            var qt = 0

            inCut[s] = true
            qVisited[qt++] = s

            while (qh < qt) {
                val v = qVisited[qh++]
                var e = head[v]
                while (e != -1) {
                    val trg = to[e]
                    if (!inCut[trg] && cap[e] - flow[e] > 1e-4f) {
                        inCut[trg] = true
                        qVisited[qt++] = trg
                    }
                    e = nxt[e]
                }
            }
            return inCut
        }
    }
}
