package qidk.fix

import android.graphics.Bitmap
import android.graphics.PointF
import android.media.FaceDetector

/**
 * Android side of Smart SD: builds the 512 x 512 working copies, runs the platform face detector and hands everything to
 * the pure planner [SmartSd]. One call = one [SmartSdPlan]; the UI drawer and the dispatcher both consume that plan.
 */
object SmartSdAnalyzer {
    private const val N = 512

    fun detectFaces(img: Bitmap): List<FaceInfo> = try {
        val rgb565 = img.copy(Bitmap.Config.RGB_565, false)
        val detector = FaceDetector(rgb565.width, rgb565.height, 5)
        val found = arrayOfNulls<FaceDetector.Face>(5)
        val n = detector.findFaces(rgb565, found)
        val mid = PointF()
        val out = ArrayList<FaceInfo>()
        for (i in 0 until n) {
            val f = found[i] ?: continue
            if (f.confidence() < 0.4f) continue
            f.getMidPoint(mid)
            out += FaceInfo(mid.x, mid.y, f.eyesDistance(), f.pose(FaceDetector.Face.EULER_Z))
        }
        out
    } catch (_: Exception) {
        emptyList()
    }

    fun plan(image: Bitmap, mask: Bitmap, userPositive: String, userNegative: String, steps: Int = 12, seed: Int = 42): SmartSdPlan {
        val img = if (image.width == N && image.height == N) image else Bitmap.createScaledBitmap(image, N, N, true)
        val msk = if (mask.width == N && mask.height == N) mask else Bitmap.createScaledBitmap(mask, N, N, false)
        val argb = IntArray(N * N).also { img.getPixels(it, 0, N, 0, 0, N, N) }
        val hole = ClassicalInpainter.holeBytes(msk)
        val scene = SmartSd.analyseScene(argb, N, N, detectFaces(img))
        val ring = SmartPrompt.analyse(argb, hole, N, N)
        val target = SmartSd.analyseTarget(scene, hole, N, N, ring)
        return SmartSd.plan(userPositive, userNegative, scene, target, steps, seed)
    }
}
