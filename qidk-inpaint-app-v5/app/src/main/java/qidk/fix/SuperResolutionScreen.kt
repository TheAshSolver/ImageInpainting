package qidk.fix

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.slider.Slider
import com.qualcomm.qidk.inpaint.R
import kotlin.math.ceil

/**
 * Process-wide memory of what is on screen, so that toggling Light / Dark (which recreates the activity) does not throw the
 * loaded picture, the mask, the last results or the selections away. Lives as long as the process, nothing is persisted.
 */
object UiSession {
    var tab = 0                                   // 0 = Inpaint, 1 = Super-Resolution
    var inpaintSource: Bitmap? = null
    var inpaintMask: Bitmap? = null
    var inpaintResult: Bitmap? = null
    var modelIndex = -1
    var backendLabel: String? = null
    var toolId = -1
    var promptPos = ""
    var promptNeg = ""
    var inpaintTelemetry: String? = null
    var plan: SmartSdPlan? = null
    var srInput: Bitmap? = null
    var srOutput: Bitmap? = null
    var srPyramid: List<Bitmap>? = null
    var srTelemetryRows: String? = null
    var srNotes: String? = null
    var srEngine = 0
    var srScale = 2
    var srBackend = 0
    var srDetail = 60f
}

/** The Super-Resolution page: engine, scale, backend, detail strength, stacked synchronised viewports and the post-run card. */
class SuperResolutionScreen(
    context: Context,
    private val engines: List<SrEngine>,
    private val sourceProvider: () -> Bitmap?
) : FrameLayout(context) {

    private val spEngine: Spinner
    private val engineNote: TextView
    private val scaleChips: ChipGroup
    private val backendChips: ChipGroup
    private val detailRow: View
    private val detailValue: TextView
    private val detailSlider: Slider
    private val scaleInfo: TextView
    private val btnUpscale: Button
    private val btnCancel: Button
    private val progressBar: ProgressBar
    private val status: TextView
    private val dual: DualViewport
    private val telemetryCard: View
    private val telemetryText: TextView
    private val notesText: TextView

    private var worker: Thread? = null
    private var progress: Progress? = null
    private val backendOrder = listOf(Backend.NPU, Backend.GPU, Backend.CPU)
    private val backendChipIds: List<Int>
    private val scaleChipIds: List<Int>

    private inner class Progress : SrProgress {
        @Volatile override var cancelled = false
        private var lastPost = 0L
        override fun report(pass: Int, passes: Int, done: Int, total: Int, note: String) {
            val now = SystemClock.uptimeMillis()
            if (now - lastPost < 120 && done != total) return
            lastPost = now
            val frac = ((pass - 1).toFloat() + done.toFloat() / max1(total)) / passes
            post {
                progressBar.progress = (frac * 1000).toInt()
                status.text = "pass $pass/$passes  ${note}"
            }
        }
        private fun max1(v: Int) = if (v < 1) 1 else v
    }

    init {
        LayoutInflater.from(context).inflate(R.layout.view_super_resolution, this, true)
        spEngine = findViewById(R.id.srEngineSpinner)
        engineNote = findViewById(R.id.srEngineNote)
        scaleChips = findViewById(R.id.srScaleChips)
        backendChips = findViewById(R.id.srBackendChips)
        detailRow = findViewById(R.id.srDetailRow)
        detailValue = findViewById(R.id.srDetailValue)
        detailSlider = findViewById(R.id.srDetailSlider)
        scaleInfo = findViewById(R.id.srScaleInfo)
        btnUpscale = findViewById(R.id.srBtnUpscale)
        btnCancel = findViewById(R.id.srBtnCancel)
        progressBar = findViewById(R.id.srProgress)
        status = findViewById(R.id.srStatus)
        dual = findViewById(R.id.srDual)
        telemetryCard = findViewById(R.id.srTelemetryCard)
        telemetryText = findViewById(R.id.srTelemetryText)
        notesText = findViewById(R.id.srNotes)
        backendChipIds = listOf(R.id.srChipNpu, R.id.srChipGpu, R.id.srChipCpu)
        scaleChipIds = listOf(R.id.srChip2x, R.id.srChip4x, R.id.srChip8x)

        val labels = engines.map { it.label + if (it.unavailableReason != null) "   (unavailable)" else "" }
        spEngine.adapter = ArrayAdapter(context, R.layout.item_spinner, labels).also { it.setDropDownViewResource(R.layout.item_spinner_dropdown) }
        spEngine.setSelection(UiSession.srEngine.coerceIn(0, engines.size - 1))
        spEngine.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) { UiSession.srEngine = pos; onEngineChanged() }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
        scaleChips.setOnCheckedStateChangeListener { _, _ -> UiSession.srScale = selectedScale(); updateScaleInfo() }
        backendChips.setOnCheckedStateChangeListener { _, _ -> UiSession.srBackend = backendOrder.indexOf(selectedBackend()) }
        detailSlider.value = UiSession.srDetail
        detailValue.text = "${UiSession.srDetail.toInt()} %"
        detailSlider.addOnChangeListener { _, v, _ -> UiSession.srDetail = v; detailValue.text = "${v.toInt()} %" }
        btnUpscale.setOnClickListener { runSelected() }
        btnCancel.setOnClickListener { progress?.cancelled = true; status.text = "cancelling after the current window..." }

        scaleChips.check(scaleChipIds[listOf(2, 4, 8).indexOf(UiSession.srScale).coerceAtLeast(0)])
        onEngineChanged()
        UiSession.srOutput?.let { out ->
            dual.setImages(UiSession.srInput, out, UiSession.srPyramid)
            showTelemetryText(UiSession.srTelemetryRows, UiSession.srNotes)
        }
    }

    private fun engine(): SrEngine = engines[spEngine.selectedItemPosition.coerceIn(0, engines.size - 1)]

    private fun selectedScale(): Int = when (scaleChips.checkedChipId) { R.id.srChip4x -> 4; R.id.srChip8x -> 8; else -> 2 }

    private fun selectedBackend(): Backend = when (backendChips.checkedChipId) { R.id.srChipGpu -> Backend.GPU; R.id.srChipCpu -> Backend.CPU; else -> Backend.NPU }

    private fun onEngineChanged() {
        val e = engine()
        val allowed = BackendMatrix.allowed(e.family)
        backendOrder.forEachIndexed { i, b -> findViewById<Chip>(backendChipIds[i]).isEnabled = b in allowed }
        if (selectedBackend() !in allowed) backendChips.check(backendChipIds[backendOrder.indexOf(allowed.first())])
        else if (UiSession.srBackend in backendOrder.indices && backendOrder[UiSession.srBackend] in allowed) backendChips.check(backendChipIds[UiSession.srBackend])
        listOf(2, 4, 8).forEachIndexed { i, s -> findViewById<Chip>(scaleChipIds[i]).isEnabled = s <= e.maxScale }
        if (selectedScale() > e.maxScale) scaleChips.check(scaleChipIds[0])
        engineNote.text = e.unavailableReason ?: ""
        engineNote.visibility = if (e.unavailableReason != null) VISIBLE else GONE
        detailRow.visibility = if (e is NeuralSrEngine || e is SdSuperResEngine) VISIBLE else GONE
        btnUpscale.isEnabled = e.unavailableReason == null && worker == null
        updateScaleInfo()
    }

    private fun updateScaleInfo() {
        val src = sourceProvider()
        val s = selectedScale()
        val side = (src?.width ?: 512) * s
        val passes = SrRun.passesFor(s)
        var windows = 0
        var w = (src?.width ?: 512)
        repeat(passes) { w *= 2; val n = ceil(w / SrWindows.CORE.toDouble()).toInt(); windows += n * n }
        val mb = side.toLong() * side * 4 / (1024 * 1024)
        val neural = engine().let { it is NeuralSrEngine || it is SdSuperResEngine }
        scaleInfo.text = "$side x $side  •  $passes pass${if (passes > 1) "es" else ""}  •  " +
            (if (neural) "${windows} windows (2 inferences each)  •  " else "") + "≈ $mb MB output"
    }

    /** Verification aid: optionally set the shared viewport, always report it (see DualViewport.describe). */
    fun debugViewport(zoom: Float?, cx: Float?, cy: Float?, reset: Boolean): String {
        if (reset) dual.shared.reset()
        else if (zoom != null || cx != null || cy != null) dual.shared.set(zoom ?: dual.shared.zoom, cx ?: dual.shared.centerX, cy ?: dual.shared.centerY)
        return dual.describe()
    }

    fun refreshSource() {
        val src = sourceProvider()
        if (UiSession.srOutput == null) dual.setImages(src, null)
        updateScaleInfo()
    }

    private fun setRunning(running: Boolean) {
        btnUpscale.isEnabled = !running && engine().unavailableReason == null
        btnCancel.visibility = if (running) VISIBLE else GONE
        progressBar.visibility = if (running) VISIBLE else GONE
        spEngine.isEnabled = !running
        scaleChips.isEnabled = !running
        if (!running) { worker = null; progress = null }
    }

    private fun runSelected() {
        val src = sourceProvider()
        if (src == null) { Toast.makeText(context, "Load an image on the Inpaint tab first", Toast.LENGTH_SHORT).show(); return }
        val e = engine()
        val backend = selectedBackend()
        val scale = selectedScale()
        BackendMatrix.require(e.family, backend)
        val strength = detailSlider.value / 100f
        (e as? NeuralSrEngine)?.detailStrength = strength
        (e as? SdSuperResEngine)?.detailStrength = strength
        val prog = Progress()
        progress = prog
        setRunning(true)
        progressBar.progress = 0
        status.text = "starting ${e.label}..."
        worker = Thread {
            val result = try { Result.success(e.runWithTelemetry(src, backend, scale, prog)) } catch (t: Throwable) { Result.failure(t) }
            val pyramid = result.getOrNull()?.let { MipPyramid.build(it.bitmap) }
            post {
                setRunning(false)
                result.fold(
                    onSuccess = { r -> show(src, r, pyramid!!) },
                    onFailure = { t ->
                        if (t is SrCancelled) status.text = "cancelled"
                        else {
                            status.text = "failed"
                            AlertDialog.Builder(context).setTitle("❌ Super-resolution error")
                                .setMessage(t.message ?: t.javaClass.simpleName).setPositiveButton("OK", null).show()
                        }
                    }
                )
            }
        }.also { it.isDaemon = true; it.start() }
    }

    private fun show(src: Bitmap, r: SrExecutionResult, pyramid: List<Bitmap>) {
        val oldPyramid = UiSession.srPyramid
        dual.setImages(src, r.bitmap, pyramid)
        oldPyramid?.forEach { if (!it.isRecycled) it.recycle() }
        UiSession.srInput = src; UiSession.srOutput = r.bitmap; UiSession.srPyramid = pyramid
        val rows = r.telemetry.rows().joinToString("\n") { (k, v) -> k.padEnd(14) + v }
        UiSession.srTelemetryRows = rows
        UiSession.srNotes = (r.notes + listOf("power: " + r.telemetry.powerNote)).joinToString("\n")
        showTelemetryText(rows, UiSession.srNotes)
        status.text = "done in ${r.telemetry.tTotalMs} ms"
    }

    private fun showTelemetryText(rows: String?, notes: String?) {
        telemetryCard.visibility = if (rows == null) GONE else VISIBLE
        telemetryText.text = rows ?: ""
        notesText.text = notes ?: ""
    }
}

/** Inpaint | Super-Resolution switch above the content of the activity. */
object SuperResolutionTab {
    /** Switches the tab from code (debug hooks); null until install() ran. */
    @Volatile var switcher: ((Boolean) -> Unit)? = null

    fun install(activity: Activity, page: View) {
        val content = activity.findViewById<ViewGroup>(android.R.id.content)
        val original = content.getChildAt(0) ?: error("setContentView must be called before SuperResolutionTab.install")
        content.removeView(original)
        val d = activity.resources.displayMetrics.density

        val group = MaterialButtonToggleGroup(activity).apply { isSingleSelection = true; isSelectionRequired = true }
        fun segment(title: String) = MaterialButton(activity, null, com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            id = View.generateViewId(); text = title; isAllCaps = false; textSize = 13f
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        val inpaint = segment("🎨 Inpaint")
        val sr = segment("⚡ Super-Resolution")
        group.addView(inpaint); group.addView(sr)

        val frame = FrameLayout(activity)
        frame.addView(original, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        frame.addView(page, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)

        fun select(srTab: Boolean) {
            UiSession.tab = if (srTab) 1 else 0
            original.visibility = if (srTab) View.GONE else View.VISIBLE
            page.visibility = if (srTab) View.VISIBLE else View.GONE
            if (srTab) (page as? SuperResolutionScreen)?.refreshSource()
        }
        group.addOnButtonCheckedListener { _, checkedId, isChecked -> if (isChecked) select(checkedId == sr.id) }
        switcher = { toSr -> activity.runOnUiThread { group.check(if (toSr) sr.id else inpaint.id) } }
        group.check(if (UiSession.tab == 1) sr.id else inpaint.id)
        select(UiSession.tab == 1)

        val root = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(ContextCompat.getColor(activity, R.color.qidk_canvas))
        }
        root.addView(group, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            setMargins((16 * d).toInt(), (8 * d).toInt(), (16 * d).toInt(), (4 * d).toInt())
        })
        root.addView(frame, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        content.addView(root, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
    }
}
