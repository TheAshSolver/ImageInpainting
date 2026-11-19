package com.qualcomm.qidk.inpaint

import android.app.Activity
import android.content.Intent
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.content.ContextCompat
import androidx.core.widget.doAfterTextChanged
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.qualcomm.qidk.inpaint.engine.ModelPreloadManager
import com.qualcomm.qidk.inpaint.engine.OnDeviceProcessDriver
import com.qualcomm.qidk.inpaint.router.RouterClassifier
import com.qualcomm.qidk.inpaint.segmentation.MediaPipeSegmenter
import com.qualcomm.qidk.inpaint.ui.InpaintCanvasView
import com.qualcomm.qidk.inpaint.utils.GrabCutEngine
import com.qualcomm.qidk.inpaint.utils.MagicTouchEngine
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import org.opencv.android.OpenCVLoader
import qidk.fix.*

class MainActivity : AppCompatActivity() {

    private lateinit var canvasView: InpaintCanvasView
    private lateinit var resultImageView: ImageView
    private lateinit var modelSpinner: Spinner
    private lateinit var backendChips: ChipGroup
    private lateinit var toolToggleGroup: MaterialButtonToggleGroup
    private val backendOrder = listOf(Backend.NPU, Backend.GPU, Backend.CPU)
    private lateinit var srEngines: List<SrEngine>
    private lateinit var srPage: SuperResolutionScreen
    private lateinit var routerBadgeText: TextView
    private lateinit var routerJustificationText: TextView
    private lateinit var preloadStatusBadge: TextView
    private lateinit var promptCard: View
    private lateinit var etPromptPositive: EditText
    private lateinit var etPromptNegative: EditText
    private lateinit var telemetryCard: View
    private lateinit var telemetryText: TextView
    private lateinit var progressBar: ProgressBar

    // Theme & Inspector HUD Views
    private lateinit var mainScrollView: View
    private lateinit var mainRootLayout: View
    private lateinit var appTitleText: TextView
    private lateinit var appSubtitleText: TextView
    private lateinit var btnThemeToggle: Button
    private lateinit var canvasContainer: View
    private lateinit var routerCard: View
    private lateinit var executionTargetCard: View
    private lateinit var pipelineTraceCard: View
    private lateinit var pipelineTraceContent: View
    private lateinit var tvTraceExpandIcon: TextView
    private lateinit var tvTraceStage1: TextView
    private lateinit var tvTraceStage2: TextView
    private lateinit var tvTraceStage3: TextView
    private lateinit var tvTraceStage4: TextView
    private lateinit var tvTraceStage5: TextView
    private lateinit var tvResultsHeader: TextView
    private var traceGeneration = 0

    private var currentBitmap: Bitmap? = null

    // Gallery Picker Contract
    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { loadBitmapFromUri(it) }
    }

    // Camera Capture Contract
    private val takePhotoLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        bitmap?.let { setCanvasImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        // Light / Dark is an AppCompat night mode (persisted); every colour of every screen is a theme token that follows it.
        AppCompatDelegate.setDefaultNightMode(getSharedPreferences(UI_PREFS, MODE_PRIVATE).getInt(KEY_NIGHT, AppCompatDelegate.MODE_NIGHT_YES))
        super.onCreate(savedInstanceState)
        try {
            System.loadLibrary("qidk_classical")
        } catch (t: Throwable) {
            Log.e("MainActivity", "Failed to load qidk_classical: ${t.message}")
        }
        if (!OpenCVLoader.initDebug()) {
            Log.e("MainActivity", "OpenCVLoader.initDebug() failed")
        } else {
            Log.i("MainActivity", "OpenCV loaded successfully")
        }
        setContentView(R.layout.activity_main)

        // View Binding
        mainScrollView = findViewById(R.id.mainScrollView)
        mainRootLayout = findViewById(R.id.mainRootLayout)
        appTitleText = findViewById(R.id.appTitleText)
        appSubtitleText = findViewById(R.id.appSubtitleText)
        btnThemeToggle = findViewById(R.id.btnThemeToggle)
        canvasContainer = findViewById(R.id.canvasContainer)
        routerCard = findViewById(R.id.routerCard)
        executionTargetCard = findViewById(R.id.executionTargetCard)
        pipelineTraceCard = findViewById(R.id.pipelineTraceCard)
        pipelineTraceContent = findViewById(R.id.pipelineTraceContent)
        tvTraceExpandIcon = findViewById(R.id.tvTraceExpandIcon)
        tvTraceStage1 = findViewById(R.id.tvTraceStage1)
        tvTraceStage2 = findViewById(R.id.tvTraceStage2)
        tvTraceStage3 = findViewById(R.id.tvTraceStage3)
        tvTraceStage4 = findViewById(R.id.tvTraceStage4)
        tvTraceStage5 = findViewById(R.id.tvTraceStage5)
        tvResultsHeader = findViewById(R.id.tvResultsHeader)

        canvasView = findViewById(R.id.inpaintCanvasView)
        resultImageView = findViewById(R.id.resultImageView)
        modelSpinner = findViewById(R.id.modelSpinner)
        backendChips = findViewById(R.id.backendChips)
        toolToggleGroup = findViewById(R.id.toolToggleGroup)
        routerBadgeText = findViewById(R.id.routerBadgeText)
        routerJustificationText = findViewById(R.id.routerJustificationText)
        preloadStatusBadge = findViewById(R.id.preloadStatusBadge)
        promptCard = findViewById(R.id.promptCard)
        etPromptPositive = findViewById(R.id.etPromptPositive)
        etPromptNegative = findViewById(R.id.etPromptNegative)
        telemetryCard = findViewById(R.id.telemetryCard)
        telemetryText = findViewById(R.id.telemetryText)
        progressBar = findViewById(R.id.progressBar)

        setupModelSpinner()
        setupButtons()
        triggerModelPreload()

        srEngines = listOf(
            NeuralSrEngine("migan_sr", "MIGAN detail transfer (NPU/GPU/CPU)", ModelFamily.SR_MIGAN, "migan"),
            NeuralSrEngine("lama_sr", "LaMa detail transfer (NPU/GPU/CPU)", ModelFamily.SR_LAMA, "lama"),
            NeuralSrEngine("aotgan_sr", "AOT-GAN detail transfer (NPU/GPU/CPU)", ModelFamily.SR_AOT_GAN, "aotgan"),
            SdSuperResEngine(),
            TarshSrEngine(),
            BicubicSrEngine()
        )
        Log.i("QIDK.SR", "native SR kernels loaded=${SrNative.available}")
        srPage = SuperResolutionScreen(this, srEngines) { canvasView.getSourceBitmap() }
        SuperResolutionTab.install(this, srPage)

        RouterRuntime.init(applicationContext)
        registerDebugHooks()

        val night = (resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK) ==
            android.content.res.Configuration.UI_MODE_NIGHT_YES
        btnThemeToggle.text = if (night) "🌙" else "☀️"

        restoreOrLoadSample()
        startPowerCalibrationOnce()
    }

    override fun onResume() {
        super.onResume()
        PowerMonitor.get().start()
    }

    override fun onPause() {
        saveSession()
        PowerMonitor.get().stop()
        super.onPause()
    }

    private fun startPowerCalibrationOnce() {
        val prefs = getSharedPreferences(UI_PREFS, MODE_PRIVATE)
        val known = prefs.getInt(KEY_POWER_SIGN, 0)
        PowerMonitor.get().loadSign = known
        if (known != 0) return
        Thread {
            val sign = PowerCalibration.measureSign(PowerMonitor.get())
            Log.i("QIDK.Power", "load sign calibrated: $sign (0 = burn not visible on the fuel gauge, retried next start)")
            if (sign != 0) prefs.edit().putInt(KEY_POWER_SIGN, sign).apply()
            PowerMonitor.get().loadSign = sign
        }.apply { isDaemon = true; start() }
    }

    private fun saveSession() {
        if (!::canvasView.isInitialized) return
        UiSession.inpaintSource = canvasView.getSourceBitmap()
        UiSession.inpaintMask = canvasView.getRawMaskBitmap()
        UiSession.modelIndex = modelSpinner.selectedItemPosition
        UiSession.backendLabel = selectedBackend().name
        UiSession.toolId = toolToggleGroup.checkedButtonId
        UiSession.promptPos = etPromptPositive.text.toString()
        UiSession.promptNeg = etPromptNegative.text.toString()
        UiSession.inpaintTelemetry = if (telemetryCard.visibility == View.VISIBLE) telemetryText.text.toString() else null
    }

    private fun restoreOrLoadSample() {
        val src = UiSession.inpaintSource
        if (src == null) { loadDefaultSample(); return }
        currentBitmap = src
        canvasView.setSourceBitmap(src)
        UiSession.inpaintMask?.let { canvasView.setRawMaskBitmap(it) }
        UiSession.inpaintResult?.let { resultImageView.setImageBitmap(it) }
        UiSession.inpaintTelemetry?.let { telemetryText.text = it; telemetryCard.visibility = View.VISIBLE }
        UiSession.plan?.takeIf { it.smart }?.let { showPlan(it) }
        if (UiSession.toolId > 0) toolToggleGroup.check(UiSession.toolId)
        updateRouterLogic()
    }

    private fun backendChip(b: Backend): Chip = findViewById(when (b) {
        Backend.NPU -> R.id.chipBackendNpu
        Backend.GPU -> R.id.chipBackendGpu
        Backend.CPU -> R.id.chipBackendCpu
    })

    private fun refreshBackends(modelText: String) {
        val family = BackendMatrix.familyOf(modelText)
        val allowed = family?.let { BackendMatrix.allowed(it) } ?: listOf(Backend.NPU)
        val prev = selectedBackend()
        for (b in backendOrder) backendChip(b).isEnabled = b in allowed
        backendChips.check(backendChip(if (prev in allowed) prev else allowed.first()).id)
    }

    private fun selectedBackend(): Backend = when (backendChips.checkedChipId) {
        R.id.chipBackendGpu -> Backend.GPU
        R.id.chipBackendCpu -> Backend.CPU
        else -> Backend.NPU
    }

    private fun setControlsEnabled(on: Boolean) {
        modelSpinner.isEnabled = on
        if (on) refreshBackends(modelSpinner.selectedItem?.toString() ?: "") else backendOrder.forEach { backendChip(it).isEnabled = false }
    }

    private fun isSmartSelected() = (modelSpinner.selectedItem?.toString() ?: "").contains("Smart")

    private fun setupModelSpinner() {
        val models = arrayOf(
            "Auto (Router Recommended)",
            "SD Duel (Fast Inpaint • 1.5s NPU)",
            "Smart SD (Prompt Inpaint • CFG 7.5 Object Insertion)",
            "LaMa Dilated (Large Voids)",
            "MIGAN (Portraits & Speed)",
            "AOT-GAN (Dense Texture)",
            "Standard SD 1.5 (Fast 4s • Untouched)",
            "Fast Marching (Telea)",
            "Fluid Streamlines (Navier-Stokes)",
            "Criminisi Exemplar",
            "Biharmonic Poisson"
        )
        val adapter = ArrayAdapter(this, R.layout.item_spinner, models)
        adapter.setDropDownViewResource(R.layout.item_spinner_dropdown)
        modelSpinner.adapter = adapter

        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = parent?.getItemAtPosition(position)?.toString() ?: ""
                refreshBackends(selected)
                val smart = selected.contains("Smart")
                val isSd = smart || selected.contains("SD") || selected.contains("Diffusion") || selected.contains("Duel") || selected.contains("Dual")
                promptCard.visibility = if (isSd) View.VISIBLE else View.GONE
                // The 5-stage drawer belongs to Smart SD only: Standard SD sends the user's words as they are, there is nothing to trace.
                pipelineTraceCard.visibility = if (smart) View.VISIBLE else View.GONE
                if (smart) updatePipelineTrace()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                promptCard.visibility = View.GONE
                pipelineTraceCard.visibility = View.GONE
            }
        }
        if (UiSession.modelIndex in models.indices) modelSpinner.setSelection(UiSession.modelIndex)
        refreshBackends(models[UiSession.modelIndex.coerceIn(0, models.size - 1)])
        UiSession.backendLabel?.let { name -> Backend.values().firstOrNull { it.name == name }?.let { b -> if (backendChip(b).isEnabled) backendChips.check(backendChip(b).id) } }
        etPromptPositive.setText(UiSession.promptPos)
        etPromptNegative.setText(UiSession.promptNeg)
        etPromptPositive.doAfterTextChanged { updatePipelineTrace() }
        etPromptNegative.doAfterTextChanged { updatePipelineTrace() }
    }

    private fun triggerModelPreload() {
        preloadStatusBadge.visibility = View.VISIBLE
        preloadStatusBadge.text = "⚡ Preloading GAN models on Hexagon NPU..."

        // 1. Synchronous GAN preload during startup
        val ganTime = ModelPreloadManager.preloadGanModelsSync(this)
        preloadStatusBadge.text = "⚡ GANs ready on NPU (${ganTime}ms) | Streaming SD models in background..."
        preloadStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.qidk_cyan))

        // 2. Asynchronous SD background streaming
        ModelPreloadManager.preloadSdModelsAsync { sdTime ->
            runOnUiThread {
                preloadStatusBadge.text = "✅ Hexagon NPU Ready (GANs: ${ganTime}ms, SD: ${sdTime}ms)"
                preloadStatusBadge.setTextColor(ContextCompat.getColor(this, R.color.qidk_emerald))
            }
        }
    }

    private fun setupButtons() {
        // 3-way segmented control: Manual Brush | Smart AI Touch | Bounding Box (Material toggle group, colours from the theme)
        toolToggleGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            when (checkedId) {
                R.id.btnModeBrush -> { canvasView.setMode(InpaintCanvasView.Mode.BRUSH); Toast.makeText(this, "🖌️ Manual Precision: Finger draw mask", Toast.LENGTH_SHORT).show() }
                R.id.btnModeSmart -> { canvasView.setMode(InpaintCanvasView.Mode.SMART); Toast.makeText(this, "✨ Smart AI Touch: Circle an object or scribble over it", Toast.LENGTH_SHORT).show() }
                R.id.btnModeBox -> { canvasView.setMode(InpaintCanvasView.Mode.BOUNDING_BOX); Toast.makeText(this, "🔲 Box Mode: Tap 1st corner, then opposite corner", Toast.LENGTH_SHORT).show() }
            }
        }
        toolToggleGroup.check(R.id.btnModeBrush)

        btnThemeToggle.setOnClickListener { toggleTheme() }

        findViewById<View>(R.id.pipelineTraceHeader).setOnClickListener {
            val willShow = pipelineTraceContent.visibility != View.VISIBLE
            pipelineTraceContent.visibility = if (willShow) View.VISIBLE else View.GONE
            tvTraceExpandIcon.text = if (willShow) "▲" else "▼"
        }

        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        canvasView.onBoxSelectionStatusChanged = { msg ->
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
        }

        canvasView.onCircleSelectionStatusChanged = { msg ->
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
        }

        canvasView.onMaskChanged = {
            updateRouterLogic()
        }

        canvasView.onSmartMessage = { msg ->
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
        }

        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            takePhotoLauncher.launch(null)
        }

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            canvasView.clearMask()
            val feedback = qidk.fix.OnlineLearningEngine.onUserFeedback(saved = false)
            if (feedback != null) Toast.makeText(this, feedback, Toast.LENGTH_SHORT).show()
            updateRouterLogic()
        }

        findViewById<Button>(R.id.btnUndo).setOnClickListener {
            canvasView.undo()
            val feedback = qidk.fix.OnlineLearningEngine.onUserFeedback(saved = false)
            if (feedback != null) Toast.makeText(this, feedback, Toast.LENGTH_SHORT).show()
            updateRouterLogic()
        }

        resultImageView.setOnLongClickListener {
            val bmp = UiSession.inpaintResult
            if (bmp != null) {
                val feedback = qidk.fix.OnlineLearningEngine.onUserFeedback(saved = true)
                Toast.makeText(this, feedback ?: "💾 Output Saved (Online Bandit Reward +1.0)", Toast.LENGTH_SHORT).show()
                true
            } else false
        }
        findViewById<Button>(R.id.btnZoomIn)?.setOnClickListener {
            canvasView.zoomIn()
        }

        findViewById<Button>(R.id.btnZoomOut)?.setOnClickListener {
            canvasView.zoomOut()
        }

        findViewById<Button>(R.id.btnZoomReset)?.setOnClickListener {
            canvasView.resetZoom()
        }

        findViewById<Button>(R.id.chipClean)?.setOnClickListener {
            etPromptPositive.setText("seamless clean background, natural texture, highly realistic")
            updatePipelineTrace()
        }
        findViewById<Button>(R.id.chipWood)?.setOnClickListener {
            etPromptPositive.setText("smooth polished wood surface, warm natural indoor lighting")
            updatePipelineTrace()
        }
        findViewById<Button>(R.id.chipNature)?.setOnClickListener {
            etPromptPositive.setText("lush green foliage and leaves, sharp natural texture")
            updatePipelineTrace()
        }
        findViewById<Button>(R.id.chipWater)?.setOnClickListener {
            etPromptPositive.setText("crystal clear water surface with gentle ripples")
            updatePipelineTrace()
        }
        findViewById<Button>(R.id.chipApple)?.setOnClickListener {
            etPromptPositive.setText("fresh crisp red apple, highly detailed, studio lighting")
            updatePipelineTrace()
        }

        findViewById<Button>(R.id.btnLoadSample)?.setOnClickListener {
            showSamplePickerDialog()
        }

        findViewById<Button>(R.id.btnRunInpaint).setOnClickListener {
            runInpainting()
        }
    }

    /** Switches the AppCompat night mode. The activity is recreated; saveSession() / restoreOrLoadSample() carry the work over. */
    private fun toggleTheme() {
        saveSession()
        val night = (resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK) ==
            android.content.res.Configuration.UI_MODE_NIGHT_YES
        val next = if (night) AppCompatDelegate.MODE_NIGHT_NO else AppCompatDelegate.MODE_NIGHT_YES
        getSharedPreferences(UI_PREFS, MODE_PRIVATE).edit().putInt(KEY_NIGHT, next).apply()
        Log.i("QIDK.Theme", "night mode -> ${if (next == AppCompatDelegate.MODE_NIGHT_YES) "dark" else "light"}")
        AppCompatDelegate.setDefaultNightMode(next)
    }

    private fun showSamplePickerDialog() {
        val internalDir = File(filesDir, "samples")
        val samplesDir = File(getExternalFilesDir(null), "samples")
        val altDir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint/files/samples")
        val v2Dir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint.v2/files/samples")
        val picDir = File("/sdcard/Pictures/Inpainting102")
        val picDirAlt = File("/sdcard/Pictures/Inpainting")
        val lamaDir = File("/data/local/tmp/lama/samples")
        val candidates = listOf(internalDir, samplesDir, altDir, v2Dir, picDir, picDirAlt, lamaDir)
        val targetDir = candidates.firstOrNull { it.exists() && it.isDirectory && (it.listFiles()?.isNotEmpty() == true) }
            ?: candidates.firstOrNull { it.exists() && it.isDirectory }

        if (targetDir == null) {
            Toast.makeText(this, "Samples directory not found on device.", Toast.LENGTH_LONG).show()
            return
        }

        val sampleFiles = targetDir.listFiles { file ->
            file.isFile && (file.name.endsWith(".png", true) || file.name.endsWith(".jpg", true)) && !file.name.contains("mask")
        }?.sortedBy { it.name } ?: emptyList()

        if (sampleFiles.isEmpty()) {
            Toast.makeText(this, "No samples found in ${targetDir.absolutePath}", Toast.LENGTH_SHORT).show()
            return
        }

        val items = sampleFiles.map { it.name }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("🧪 Select Benchmark Sample (${sampleFiles.size} available)")
            .setItems(items) { _, which ->
                val selectedFile = sampleFiles[which]
                loadSampleFile(selectedFile, targetDir)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun loadSampleFile(imageFile: File, baseDir: File) {
        try {
            val bmp = BitmapFactory.decodeFile(imageFile.absolutePath)
            if (bmp != null) {
                setCanvasImage(bmp)
                canvasView.clearMask(saveUndo = false)

                // Check for companion mask
                val stem = imageFile.nameWithoutExtension
                val maskCandidates = arrayOf(
                    File(baseDir, "masks/${imageFile.name}"),
                    File(baseDir, "masks/${stem}.png"),
                    File(baseDir, "${stem}_mask.png"),
                    File(baseDir, "${stem}_mask.jpg")
                )
                var foundMask: File? = null
                for (cand in maskCandidates) {
                    if (cand.exists()) {
                        foundMask = cand
                        break
                    }
                }

                if (foundMask != null) {
                    val maskBmp = BitmapFactory.decodeFile(foundMask.absolutePath)
                    if (maskBmp != null) {
                        canvasView.applyGrabCutMask(maskBmp)
                        updateRouterLogic()
                        Toast.makeText(this, "Loaded ${imageFile.name} + Mask", Toast.LENGTH_SHORT).show()
                        return
                    }
                }

                updateRouterLogic()
                Toast.makeText(this, "Loaded ${imageFile.name}", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading sample: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadDefaultSample() {
        val internalDir = File(filesDir, "samples")
        val samplesDir = File(getExternalFilesDir(null), "samples")
        val altDir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint/files/samples")
        val v2Dir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint.v2/files/samples")
        val picDir = File("/sdcard/Pictures/Inpainting102")
        val picDirAlt = File("/sdcard/Pictures/Inpainting")
        val lamaDir = File("/data/local/tmp/lama/samples")
        val candidates = listOf(internalDir, samplesDir, altDir, v2Dir, picDir, picDirAlt, lamaDir)
        val targetDir = candidates.firstOrNull { it.exists() && it.isDirectory && File(it, "001.png").exists() }
            ?: candidates.firstOrNull { it.exists() && it.isDirectory && (it.listFiles()?.isNotEmpty() == true) }

        if (targetDir != null) {
            val s001 = File(targetDir, "001.png")
            if (s001.exists() && s001.canRead()) {
                try {
                    val testBmp = BitmapFactory.decodeFile(s001.absolutePath)
                    if (testBmp != null) {
                        loadSampleFile(s001, targetDir)
                        return
                    }
                } catch (_: Exception) {}
            }
        }

        try {
            val stream = assets.open("sample.png")
            val bmp = BitmapFactory.decodeStream(stream)
            stream.close()
            if (bmp != null) {
                setCanvasImage(bmp)
                return
            }
        } catch (e: Exception) {
            // fallback if asset not present
        }
        val bmp = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        bmp.eraseColor(Color.rgb(180, 180, 180))
        setCanvasImage(bmp)
    }

    private fun setCanvasImage(bitmap: Bitmap) {
        val scaled = Bitmap.createScaledBitmap(bitmap, 512, 512, true)
        currentBitmap = scaled
        canvasView.setSourceBitmap(scaled)
        updateRouterLogic()
    }

    private fun loadBitmapFromUri(uri: Uri) {
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            bitmap?.let { setCanvasImage(it) }
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading image: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateRouterLogic() {
        val src = canvasView.getSourceBitmap() ?: return
        val mask = canvasView.getMaskBitmap()

        val decision = RouterClassifier.classifyAndRoute(src, mask)

        val color = ContextCompat.getColor(this, when (decision.recommendedModel) {
            "MIGAN" -> R.color.qidk_emerald
            "AOTGAN" -> R.color.qidk_cyan
            else -> R.color.qidk_purple
        })

        routerBadgeText.setTextColor(color)
        routerBadgeText.text = "🎯 RECOMMENDED: ${decision.recommendedModel} (${decision.ruleTriggered})"
        routerJustificationText.text = decision.justification

        updatePipelineTrace()
    }

    /** Smart SD only. The plan is computed off the UI thread; a stale result (older generation) is dropped. */
    private fun updatePipelineTrace() {
        if (!isSmartSelected()) return
        val src = canvasView.getSourceBitmap() ?: return
        val mask = canvasView.getMaskBitmap()
        val pos = etPromptPositive.text.toString()
        val neg = etPromptNegative.text.toString()
        val gen = ++traceGeneration
        Thread {
            val plan = try { SmartSdAnalyzer.plan(src, mask, pos, neg) } catch (e: Exception) { Log.w("QIDK.SD", "plan failed: ${e.message}"); null }
            runOnUiThread { if (gen == traceGeneration && plan != null) showPlan(plan) }
        }.apply { isDaemon = true; start() }
    }

    /** The drawer shows exactly the strings the runner will receive (see OnDeviceProcessDriver: it uses the same plan). */
    private fun showPlan(p: SmartSdPlan) {
        UiSession.plan = p
        findViewById<TextView>(R.id.tvTraceHeaderTitle).text = String.format(java.util.Locale.US, "🧠 Smart SD pipeline  •  CFG %.1f  •  %d steps", p.cfg, p.steps)
        tvTraceStage1.text = p.stage1
        tvTraceStage2.text = p.stage2
        tvTraceStage3.text = p.stage3
        tvTraceStage4.text = p.positive
        tvTraceStage5.text = p.negative
        Log.i("QIDK.SD", "plan intent=${p.intent} cfg=${p.cfg} pos='${p.positive}' neg='${p.negative}'")
    }

    private fun runInpainting() {
        val src = canvasView.getSourceBitmap()
        val mask = canvasView.getMaskBitmap()

        if (src == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
            return
        }

        // Check if user has painted or selected a mask
        var hasMask = false
        val maskPixels = IntArray(mask.width * mask.height)
        mask.getPixels(maskPixels, 0, mask.width, 0, 0, mask.width, mask.height)
        for (c in maskPixels) {
            if (Color.red(c) > 128) {
                hasMask = true
                break
            }
        }
        if (!hasMask) {
            Toast.makeText(this, "Please draw a mask with Brush or select a Box first", Toast.LENGTH_SHORT).show()
            return
        }

        val btnRunInpaint = findViewById<Button>(R.id.btnRunInpaint)
        btnRunInpaint.isEnabled = false
        setControlsEnabled(false)
        progressBar.visibility = View.VISIBLE

        val selectedOption = modelSpinner.selectedItem?.toString() ?: "Auto (Router Recommended)"
        val backend = selectedBackend()

        val targetModel = when {
            selectedOption.contains("Telea") || selectedOption.contains("Fast Marching") -> "TELEA"
            selectedOption.contains("Navier") || selectedOption.contains("Streamlines") -> "NAVIER_STOKES"
            selectedOption.contains("Criminisi") || selectedOption.contains("Exemplar") -> "CRIMINISI"
            selectedOption.contains("Biharmonic") || selectedOption.contains("Poisson") -> "BIHARMONIC"
            selectedOption.contains("Hybrid") || selectedOption.contains("Auto") -> {
                val dec = RouterClassifier.classifyAndRoute(src, mask)
                dec.recommendedModel
            }
            selectedOption.contains("MIGAN") -> "MIGAN"
            selectedOption.contains("LaMa") -> "LAMA"
            selectedOption.contains("AOT") -> "AOTGAN"
            selectedOption.contains("Smart") -> "SMART_SD"
            selectedOption.contains("Duel") || selectedOption.contains("Dual") -> "SD_DUEL"
            selectedOption.contains("SD") || selectedOption.contains("Diffusion") -> "SD"
            else -> "LAMA"
        }

        val posPrompt = etPromptPositive.text.toString().trim()
        val negPrompt = etPromptNegative.text.toString().trim()
        val isSmartPrompt = selectedOption.contains("Smart")        // Smart SD is its own model entry: no extra switch

        Thread {
            try {
                // ONE plan per run: Standard SD = the user's words untouched, Smart SD = SmartSd.plan. The drawer shows it before the NPU starts.
                val isSdModel = targetModel == "SD" || targetModel == "SMART_SD" || targetModel == "SD_DUEL"
                val plan: SmartSdPlan? = if (isSdModel) {
                    if (isSmartPrompt || targetModel == "SMART_SD") SmartSdAnalyzer.plan(src, mask, posPrompt, negPrompt)
                    else SmartSd.standardPlan(posPrompt, negPrompt, if (targetModel == "SD_DUEL") 3 else 12)
                } else null
                if (plan != null && plan.smart) runOnUiThread { showPlan(plan) }
                val telemetry = OnDeviceProcessDriver.executeInference(
                    src, mask, targetModel, backend, posPrompt, negPrompt, smartPrompt = isSmartPrompt, plan = plan
                )
                val predEst = qidk.fix.PredictionEngine.predict(targetModel, 0.15f, 0.05f, telemetry.startTempC)
                qidk.fix.AdaptiveDriftTracker.recordInference(predEst.expectedLatencyMs, telemetry.latencyMs.toFloat())
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnRunInpaint.isEnabled = true
                    setControlsEnabled(true)
                    resultImageView.setImageBitmap(telemetry.resultBitmap)
                    UiSession.inpaintResult = telemetry.resultBitmap
                    telemetryCard.visibility = View.VISIBLE

                    telemetry.smartTrace?.let { trace ->
                        tvTraceStage1.text = trace.stage1FullImage
                        tvTraceStage2.text = trace.stage2MaskedContext
                        tvTraceStage3.text = trace.stage3UserInput
                        tvTraceStage4.text = trace.stage4FinalPositive
                        tvTraceStage5.text = trace.stage5Negative
                        pipelineTraceCard.visibility = View.VISIBLE
                        pipelineTraceContent.visibility = View.VISIBLE
                        tvTraceExpandIcon.text = "▲"
                    }

                    val run = telemetry.run
                    telemetryText.typeface = Typeface.MONOSPACE
                    telemetryText.text = buildString {
                        append("Model".padEnd(14)).append(telemetry.modelName).append('\n')
                        if (run != null) {
                            run.rows().forEach { (k, v) -> append(k.padEnd(14)).append(v).append('\n') }
                            append("power note".padEnd(14)).append(run.powerNote)
                        } else {
                            append("Latency".padEnd(14)).append("${telemetry.latencyMs} ms")
                        }
                    }
                    Toast.makeText(this, "✨ Inpainting complete with ${telemetry.modelName}", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnRunInpaint.isEnabled = true
                    setControlsEnabled(true)
                    AlertDialog.Builder(this)
                        .setTitle("❌ Inference Error")
                        .setMessage(e.message ?: "An unexpected error occurred during inference")
                        .setPositiveButton("OK", null)
                        .show()
                }
            }
        }.start()
    }

    private fun findSample(name: String): Pair<File, File>? {
        val dirs = listOf(
            File(filesDir, "samples"),
            File(getExternalFilesDir(null), "samples"),
            File("/sdcard/Android/data/$packageName/files/samples"),
            File("/sdcard/Pictures/Inpainting102"),
            File("/data/local/tmp/lama/samples")
        )
        for (d in dirs) {
            val f = File(d, "$name.png")
            if (f.exists()) return f to d
            val fNoExt = File(d, name)
            if (fNoExt.exists()) return fNoExt to d
        }
        return null
    }

    private fun registerDebugHooks() {
        DebugHooks.install(this, mapOf(
            "qidk.debug.MODE" to { i ->
                val m = i.getStringExtra("mode") ?: "brush"
                canvasView.setMode(when (m) {
                    "smart" -> InpaintCanvasView.Mode.SMART
                    "box" -> InpaintCanvasView.Mode.BOUNDING_BOX
                    else -> InpaintCanvasView.Mode.BRUSH
                })
                canvasView.logGeometry()
                "mode=$m"
            },
            "qidk.debug.SAMPLE" to { i ->
                val name = i.getStringExtra("name") ?: "001"
                val withMask = i.getBooleanExtra("mask", false)
                val s = findSample(name)
                if (s == null) "sample $name not found" else {
                    runOnUiThread {
                        loadSampleFile(s.first, s.second)
                        if (!withMask) canvasView.clearMask(saveUndo = false)
                    }
                    "loaded $name mask=$withMask"
                }
            },
            "qidk.debug.STROKE" to { i ->
                DebugHooks.injectStroke(canvasView, DebugHooks.parsePoints(i.getStringExtra("pts")))
            },
            "qidk.debug.CLASSICAL" to { i ->
                val algoText = i.getStringExtra("algo") ?: "telea"
                val algo = ClassicalAlgo.fromText(algoText)
                val src = canvasView.getSourceBitmap()
                val hole = if (src != null) ClassicalInpainter.holeBytes(canvasView.getMaskBitmap()) else null
                if (algo == null || src == null || hole == null) "bad algo or no image" else DebugHooks.async("qidk.debug.CLASSICAL") {
                    val out = ClassicalInpainter.run(src, hole, algo)
                    val f = File(getExternalFilesDir(null), "out/classical_$algoText.png")
                    f.parentFile?.mkdirs()
                    f.outputStream().use { out.bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    "used=${out.used} ms=${out.ms} out=${f.absolutePath}"
                }
            },
            "qidk.debug.SD" to { i ->
                val src = canvasView.getSourceBitmap()
                val mask = canvasView.getMaskBitmap()
                if (src == null) "no image" else DebugHooks.async("qidk.debug.SD") {
                    val smart = (i.getStringExtra("mode") ?: "normal") == "smart"
                    val t = OnDeviceProcessDriver.executeInference(
                        src, mask, "SD", Backend.NPU,
                        i.getStringExtra("prompt") ?: "", i.getStringExtra("neg") ?: "",
                        smartPrompt = smart, seed = i.getIntExtra("seed", 42)
                    )
                    val f = File(getExternalFilesDir(null), "out/sd_${if (smart) "smart" else "normal"}_${i.getStringExtra("tag") ?: "x"}.png")
                    f.parentFile?.mkdirs()
                    f.outputStream().use { t.resultBitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    val sha = MessageDigest.getInstance("SHA-256").digest(f.readBytes()).joinToString("") { "%02x".format(it) }.take(16)
                    "out=${f.absolutePath} sha256=$sha ms=${t.latencyMs}"
                }
            },
            "qidk.debug.SR" to { i ->
                DebugHooks.async("qidk.debug.SR") {
                    val id = i.getStringExtra("engine") ?: "bicubic"
                    val backend = Backend.values().firstOrNull { it.name == (i.getStringExtra("backend") ?: "CPU") } ?: Backend.CPU
                    val scale = i.getIntExtra("scale", 2)
                    val srcPath = i.getStringExtra("src")
                    val src = if (srcPath != null) BitmapFactory.decodeFile(srcPath) else canvasView.getSourceBitmap()
                    val engine = srEngines.firstOrNull { it.id == id }
                    if (src != null) i.getStringExtra("in")?.let { p ->        // the exact input the engine gets, for tools/sr_metrics.py
                        File(p).apply { parentFile?.mkdirs() }.outputStream().use { src.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    }
                    if (src == null || engine == null) "no source or unknown engine $id"
                    else if (engine.unavailableReason != null) "engine disabled: ${engine.unavailableReason}"
                    else {
                        BackendMatrix.require(engine.family, backend)
                        val res = engine.runWithTelemetry(src, backend, scale)
                        val out = res.bitmap
                        val telem = res.telemetry
                        i.getStringExtra("out")?.let { p ->
                            File(p).apply { parentFile?.mkdirs() }.outputStream().use { out.compress(Bitmap.CompressFormat.PNG, 100, it) }
                        }
                        i.getStringExtra("detail")?.toFloatOrNull()?.let { d -> (engine as? NeuralSrEngine)?.detailStrength = d; (engine as? SdSuperResEngine)?.detailStrength = d }
                        res.notes.forEach { Log.i("QIDK.SR", "note: $it") }
                        val line = "engine=$id backend=$backend scale=${scale}x out=${out.width}x${out.height} ms=${telem.tTotalMs} prep=${telem.tPrepMs} infer=${telem.tInferMs} post=${telem.tPostMs} " +
                            "dT=${telem.deltaTempC} W=${telem.activeWatts} J=${telem.batteryJoules} note='${telem.powerNote}'"
                        Log.i("QIDK.SR", line)
                        line
                    }
                }
            },
            "qidk.debug.INPAINT" to { i ->
                // every GAN / classical model through the real driver: qidk.debug.INPAINT --es model migan|lama|aotgan|telea --es backend NPU|GPU|CPU --es out <png>
                val src = canvasView.getSourceBitmap()
                val mask = canvasView.getMaskBitmap()
                if (src == null) "no image" else DebugHooks.async("qidk.debug.INPAINT") {
                    val model = (i.getStringExtra("model") ?: "migan").uppercase()
                    val backend = Backend.values().firstOrNull { it.name == (i.getStringExtra("backend") ?: "NPU") } ?: Backend.NPU
                    val t = OnDeviceProcessDriver.executeInference(src, mask, model, backend)
                    i.getStringExtra("out")?.let { p ->
                        File(p).apply { parentFile?.mkdirs() }.outputStream().use { t.resultBitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    }
                    val r = t.run
                    "model=${t.modelName} backend=$backend ms=${t.latencyMs} prep=${r?.tPrepMs} infer=${r?.tInferMs} post=${r?.tPostMs} total=${r?.tTotalMs} " +
                        "dT=${r?.deltaTempC} W=${r?.activeWatts} J=${r?.batteryJoules} note='${r?.powerNote}'"
                }
            },
            "qidk.debug.TAB" to { i ->
                val toSr = (i.getStringExtra("tab") ?: "sr") == "sr"
                SuperResolutionTab.switcher?.invoke(toSr)
                "tab=${if (toSr) "sr" else "inpaint"}"
            },
            "qidk.debug.VIEWPORT" to { i ->
                var r = "sr page not ready"
                val latch = java.util.concurrent.CountDownLatch(1)
                runOnUiThread {
                    r = srPage.debugViewport(i.getStringExtra("zoom")?.toFloatOrNull(), i.getStringExtra("cx")?.toFloatOrNull(),
                        i.getStringExtra("cy")?.toFloatOrNull(), i.getBooleanExtra("reset", false))
                    latch.countDown()
                }
                latch.await(2, java.util.concurrent.TimeUnit.SECONDS)
                Log.i("QIDK.Viewport", r)
                r
            },
            "qidk.debug.THEMEINFO" to { _ ->
                val night = (resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK) == android.content.res.Configuration.UI_MODE_NIGHT_YES
                fun c(id: Int) = ContextCompat.getColor(this, id)
                fun hex(id: Int) = "#%06X".format(c(id) and 0xFFFFFF)
                fun contrast(a: Int, b: Int) = androidx.core.graphics.ColorUtils.calculateContrast(c(a), c(b))
                val src = canvasView.getSourceBitmap()
                val hash = if (src == null) "none" else {
                    val px = IntArray(src.width * src.height).also { src.getPixels(it, 0, src.width, 0, 0, src.width, src.height) }
                    val md = MessageDigest.getInstance("SHA-256")
                    for (v in px) { md.update((v shr 16).toByte()); md.update((v shr 8).toByte()); md.update(v.toByte()) }
                    md.digest().joinToString("") { "%02x".format(it) }.take(16)
                }
                val line = "night=$night canvas=${hex(R.color.qidk_canvas)} card=${hex(R.color.qidk_card)} text=${hex(R.color.qidk_text)} " +
                    "contrast_text=%.1f contrast_muted=%.1f contrast_cyan=%.1f contrast_emerald=%.1f srcHash=$hash".format(
                        contrast(R.color.qidk_text, R.color.qidk_card), contrast(R.color.qidk_text_muted, R.color.qidk_card),
                        contrast(R.color.qidk_cyan, R.color.qidk_card), contrast(R.color.qidk_emerald, R.color.qidk_card))
                Log.i("QIDK.Theme", line)
                line
            },
            "qidk.debug.PLAN" to { i ->
                // prints the five Smart-SD stages for the loaded image + mask without touching the NPU
                val src = canvasView.getSourceBitmap()
                if (src == null) "no image" else DebugHooks.async("qidk.debug.PLAN") {
                    val p = if (i.getBooleanExtra("smart", true))
                        SmartSdAnalyzer.plan(src, canvasView.getMaskBitmap(), i.getStringExtra("prompt") ?: "", i.getStringExtra("neg") ?: "")
                    else SmartSd.standardPlan(i.getStringExtra("prompt") ?: "", i.getStringExtra("neg") ?: "", 12)
                    Log.i("QIDK.SD", "S1=${p.stage1}"); Log.i("QIDK.SD", "S2=${p.stage2}"); Log.i("QIDK.SD", "S3=${p.stage3}")
                    Log.i("QIDK.SD", "S4=${p.positive}"); Log.i("QIDK.SD", "S5=${p.negative}"); Log.i("QIDK.SD", p.runParams())
                    "smart=${p.smart} cfg=${p.cfg} intent=${p.intent} S4=${p.positive}"
                }
            },
            "qidk.debug.THEME" to { i ->
                val want = i.getStringExtra("mode") ?: "toggle"
                runOnUiThread {
                    val night = (resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK) == android.content.res.Configuration.UI_MODE_NIGHT_YES
                    if (want == "toggle" || (want == "dark") != night) toggleTheme()
                }
                "theme request: $want"
            },
            "qidk.debug.FEATURES" to { i ->
                DebugHooks.async("qidk.debug.FEATURES") {
                    val names = (i.getStringExtra("names") ?: "").split(',').filter { it.isNotBlank() }
                    val dir = File(i.getStringExtra("dir") ?: "/sdcard/Android/data/$packageName/files/samples")
                    val out = File(getExternalFilesDir(null), "out/features_app.csv").apply { parentFile?.mkdirs() }
                    val sb = StringBuilder("image," + names.joinToString(",") + "\n")
                    val files = (dir.listFiles { f -> f.name.endsWith(".png") } ?: emptyArray()).sortedBy { it.name }
                    for (f in files) {
                        val img = BitmapFactory.decodeFile(f.absolutePath)?.let { Bitmap.createScaledBitmap(it, 512, 512, true) } ?: continue
                        val mf = File(dir, "masks/${f.name}")
                        val m = (if (mf.exists()) BitmapFactory.decodeFile(mf.absolutePath) else null)?.let { Bitmap.createScaledBitmap(it, 512, 512, false) } ?: continue
                        val argb = IntArray(512 * 512).also { img.getPixels(it, 0, 512, 0, 0, 512, 512) }
                        val grid = ProxyGrid.build(argb, ClassicalInpainter.holeBytes(m), 512, 512)
                        sb.append(f.name).append(',').append(names.joinToString(",") { n -> RouterFeatures.registry[n.lowercase()]?.invoke(grid)?.toString() ?: "nan" }).append('\n')
                    }
                    out.writeText(sb.toString())
                    "rows=${files.size} out=${out.absolutePath}"
                }
            },
            "qidk.debug.ROUTE" to { i ->
                val src = canvasView.getSourceBitmap()
                if (src == null) "no image" else {
                    val d = RouterClassifier.classifyAndRoute(src, canvasView.getMaskBitmap())
                    val policy = RouterRuntime.policy
                    if (policy == null) "unusable: see QIDK.Router at start-up" else {
                        val argb = IntArray(512 * 512).also { src.getPixels(it, 0, 512, 0, 0, 512, 512) }
                        val grid = ProxyGrid.build(argb, ClassicalInpainter.holeBytes(canvasView.getMaskBitmap()), 512, 512)
                        val n = i.getIntExtra("n", 1000)
                        var feat = 0L
                        var pol = 0L
                        repeat(n) { policy.decide(grid)?.let { x -> feat += x.featureMicros; pol += x.policyMicros } }
                        Log.i("QIDK.Router", "mode=${d.ruleTriggered} bench n=$n avg_feat_us=${feat / n} avg_policy_us=${pol / n}")
                        "model=${d.recommendedModel} rule=${d.ruleTriggered} avg_feat_us=${feat / n} avg_policy_us=${pol / n}"
                    }
                }
            }
        ))
    }

    companion object {
        private const val UI_PREFS = "qidk_ui"
        private const val KEY_NIGHT = "night_mode"
        private const val KEY_POWER_SIGN = "power_load_sign"
    }

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        canvasView.invalidate()
    }
}
