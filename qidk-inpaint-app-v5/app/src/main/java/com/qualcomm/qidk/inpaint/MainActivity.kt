package com.qualcomm.qidk.inpaint

import android.app.Activity
import android.content.Intent
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
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
    private lateinit var backendSpinner: Spinner
    val runtimeSpinner: Spinner get() = backendSpinner
    private lateinit var srEngines: List<SrEngine>
    private lateinit var routerBadgeText: TextView
    private lateinit var routerJustificationText: TextView
    private lateinit var preloadStatusBadge: TextView
    private lateinit var promptCard: View
    private lateinit var etPromptPositive: EditText
    private lateinit var etPromptNegative: EditText
    private lateinit var telemetryCard: View
    private lateinit var telemetryText: TextView
    private lateinit var progressBar: ProgressBar

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

        canvasView = findViewById(R.id.inpaintCanvasView)
        resultImageView = findViewById(R.id.resultImageView)
        modelSpinner = findViewById(R.id.modelSpinner)
        backendSpinner = findViewById(R.id.backendSpinner)
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
            BicubicSrEngine(),
            TarshSrEngine(),
            UnavailableSrEngine("migan_sr", "MIGAN 2x Checkerboard Inpainting", ModelFamily.SR_MIGAN, "benchmark recipe not ported (Phase 5.3)"),
            UnavailableSrEngine("lama_sr", "LaMa 2x Inpainting", ModelFamily.SR_LAMA, "benchmark recipe not ported (Phase 5.3)")
        )
        val srPage = SuperResolutionView(this, srEngines) { canvasView.getSourceBitmap() }
        SuperResolutionTab.install(this, srPage)

        RouterRuntime.init(applicationContext)
        registerDebugHooks()

        if (savedInstanceState == null) {
            loadDefaultSample()
        }
    }

    private fun refreshBackends(modelText: String) {
        val family = BackendMatrix.familyOf(modelText)
        val allowed = family?.let { BackendMatrix.allowed(it) } ?: listOf(Backend.NPU)
        val prev = backendSpinner.selectedItem?.toString()
        backendSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, allowed.map { it.label })
        allowed.indexOfFirst { it.label == prev }.takeIf { it >= 0 }?.let { backendSpinner.setSelection(it) }
        backendSpinner.isEnabled = allowed.size > 1
    }

    private fun selectedBackend(): Backend = Backend.values().firstOrNull { it.label == backendSpinner.selectedItem?.toString() } ?: Backend.NPU

    private fun setupModelSpinner() {
        val models = arrayOf(
            "Auto (Router Recommended)",
            "LaMa Dilated (Large Voids)",
            "MIGAN (Portraits & Speed)",
            "AOT-GAN (Dense Texture)",
            "Stable Diffusion 1.5 (Inpainting - DPM 12-step)",
            "Smart SD",
            "Fast Marching (Telea)",
            "Fluid Streamlines (Navier-Stokes)",
            "Criminisi Exemplar",
            "Biharmonic Poisson"
        )
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, models)
        modelSpinner.adapter = adapter

        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = parent?.getItemAtPosition(position)?.toString() ?: ""
                refreshBackends(selected)
                val isSd = selected.contains("Stable Diffusion") || selected.contains("Diffusion") || selected.contains("Smart SD")
                promptCard.visibility = if (isSd) View.VISIBLE else View.GONE
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                promptCard.visibility = View.GONE
            }
        }
        refreshBackends(models[0])
    }

    private fun triggerModelPreload() {
        preloadStatusBadge.visibility = View.VISIBLE
        preloadStatusBadge.text = "⚡ Preloading GAN models on Hexagon NPU..."

        // 1. Synchronous GAN preload during startup
        val ganTime = ModelPreloadManager.preloadGanModelsSync(this)
        preloadStatusBadge.text = "⚡ GANs ready on NPU (${ganTime}ms) | Streaming SD models in background..."
        preloadStatusBadge.setTextColor(Color.parseColor("#38BDF8"))

        // 2. Asynchronous SD background streaming
        ModelPreloadManager.preloadSdModelsAsync { sdTime ->
            runOnUiThread {
                preloadStatusBadge.text = "✅ Hexagon NPU Ready (GANs: ${ganTime}ms, SD: ${sdTime}ms)"
                preloadStatusBadge.setTextColor(Color.parseColor("#10B981"))
            }
        }
    }

    private fun setupButtons() {
        val btnModeSmart = findViewById<Button>(R.id.btnModeSmart)
        val btnModeBrush = findViewById<Button>(R.id.btnModeBrush)
        val btnModeBox = findViewById<Button>(R.id.btnModeBox)
        val btnSegGrabCut = findViewById<Button>(R.id.btnSegGrabCut)
        val btnSegAiSam = findViewById<Button>(R.id.btnSegAiSam)
        val btnSegExact = findViewById<Button>(R.id.btnSegExact)

        fun updateModeButtons(activeMode: InpaintCanvasView.Mode) {
            btnModeSmart?.backgroundTintList = ColorStateList.valueOf(
                if (activeMode == InpaintCanvasView.Mode.SMART) Color.parseColor("#0E7490") else Color.parseColor("#4B5563")
            )
            btnModeBrush?.backgroundTintList = ColorStateList.valueOf(
                if (activeMode == InpaintCanvasView.Mode.BRUSH) Color.parseColor("#2563EB") else Color.parseColor("#4B5563")
            )
            btnModeBox?.backgroundTintList = ColorStateList.valueOf(
                if (activeMode == InpaintCanvasView.Mode.BOUNDING_BOX) Color.parseColor("#0891B2") else Color.parseColor("#4B5563")
            )
        }

        fun updateEngineButtons(activeEngine: InpaintCanvasView.SegmentationEngine) {
            btnSegGrabCut?.backgroundTintList = ColorStateList.valueOf(
                if (activeEngine == InpaintCanvasView.SegmentationEngine.AUTO_GRABCUT) Color.parseColor("#2563EB") else Color.parseColor("#374151")
            )
            btnSegAiSam?.backgroundTintList = ColorStateList.valueOf(
                if (activeEngine == InpaintCanvasView.SegmentationEngine.AI_MAGIC_TOUCH) Color.parseColor("#7C3AED") else Color.parseColor("#374151")
            )
            btnSegExact?.backgroundTintList = ColorStateList.valueOf(
                if (activeEngine == InpaintCanvasView.SegmentationEngine.EXACT_SELECT) Color.parseColor("#059669") else Color.parseColor("#374151")
            )
        }

        updateModeButtons(InpaintCanvasView.Mode.SMART)
        updateEngineButtons(InpaintCanvasView.SegmentationEngine.AUTO_GRABCUT)

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

        btnModeSmart?.setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.SMART)
            updateModeButtons(InpaintCanvasView.Mode.SMART)
            Toast.makeText(this, "Smart AI Touch: circle an object or scribble over it", Toast.LENGTH_SHORT).show()
        }

        btnModeBrush?.setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.BRUSH)
            updateModeButtons(InpaintCanvasView.Mode.BRUSH)
            Toast.makeText(this, "🖌️ Manual Precision: Finger draw mask", Toast.LENGTH_SHORT).show()
        }

        btnModeBox?.setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.BOUNDING_BOX)
            updateModeButtons(InpaintCanvasView.Mode.BOUNDING_BOX)
            Toast.makeText(this, "🔲 Box Mode: Tap 1st corner, then opposite corner", Toast.LENGTH_SHORT).show()
        }

        btnSegGrabCut?.setOnClickListener {
            canvasView.setSegmentationEngine(InpaintCanvasView.SegmentationEngine.AUTO_GRABCUT)
            canvasView.setMode(InpaintCanvasView.Mode.CIRCLE_SELECT)
            updateModeButtons(InpaintCanvasView.Mode.CIRCLE_SELECT)
            updateEngineButtons(InpaintCanvasView.SegmentationEngine.AUTO_GRABCUT)
            Toast.makeText(this, "⚡ Auto-GrabCut: 4-State Energy Minimization Graph-Cut (<50ms)", Toast.LENGTH_SHORT).show()
        }

        btnSegAiSam?.setOnClickListener {
            canvasView.setSegmentationEngine(InpaintCanvasView.SegmentationEngine.AI_MAGIC_TOUCH)
            canvasView.setMode(InpaintCanvasView.Mode.CIRCLE_SELECT)
            updateModeButtons(InpaintCanvasView.Mode.CIRCLE_SELECT)
            updateEngineButtons(InpaintCanvasView.SegmentationEngine.AI_MAGIC_TOUCH)
            Toast.makeText(this, "🤖 AI Magic Touch: Promptable Vision AI Segmentation", Toast.LENGTH_SHORT).show()
        }

        btnSegExact?.setOnClickListener {
            canvasView.setSegmentationEngine(InpaintCanvasView.SegmentationEngine.EXACT_SELECT)
            canvasView.setMode(InpaintCanvasView.Mode.CIRCLE_SELECT)
            updateModeButtons(InpaintCanvasView.Mode.CIRCLE_SELECT)
            updateEngineButtons(InpaintCanvasView.SegmentationEngine.EXACT_SELECT)
            Toast.makeText(this, "🎯 Exact Area: Mask covers precisely what you circle", Toast.LENGTH_SHORT).show()
        }

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            canvasView.clearMask()
            updateRouterLogic()
        }

        findViewById<Button>(R.id.btnUndo).setOnClickListener {
            canvasView.undo()
            updateRouterLogic()
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
        }
        findViewById<Button>(R.id.chipWood)?.setOnClickListener {
            etPromptPositive.setText("smooth polished wood surface, warm natural indoor lighting")
        }
        findViewById<Button>(R.id.chipNature)?.setOnClickListener {
            etPromptPositive.setText("lush green foliage and leaves, sharp natural texture")
        }
        findViewById<Button>(R.id.chipWater)?.setOnClickListener {
            etPromptPositive.setText("crystal clear water surface with gentle ripples")
        }
        findViewById<Button>(R.id.chipApple)?.setOnClickListener {
            etPromptPositive.setText("fresh crisp red apple, highly detailed, studio lighting")
        }

        findViewById<Button>(R.id.btnLoadSample)?.setOnClickListener {
            showSamplePickerDialog()
        }

        findViewById<Button>(R.id.btnRunInpaint).setOnClickListener {
            runInpainting()
        }
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

        val color = when (decision.recommendedModel) {
            "MIGAN" -> Color.parseColor("#10B981")
            "AOTGAN" -> Color.parseColor("#3B82F6")
            else -> Color.parseColor("#8B5CF6")
        }

        routerBadgeText.setTextColor(color)
        routerBadgeText.text = "🎯 RECOMMENDED: ${decision.recommendedModel} (${decision.ruleTriggered})"
        routerJustificationText.text = decision.justification
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
        modelSpinner.isEnabled = false
        backendSpinner.isEnabled = false
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
            selectedOption.contains("Stable Diffusion") || selectedOption.contains("Diffusion") || selectedOption.contains("Smart SD") -> "SD"
            else -> "LAMA"
        }

        val posPrompt = etPromptPositive.text.toString().trim()
        val negPrompt = etPromptNegative.text.toString().trim()
        val swSmartPrompt = findViewById<Switch>(R.id.swSmartPrompt)
        val isSmartPrompt = (swSmartPrompt?.isChecked == true) || selectedOption.contains("Smart")

        Thread {
            try {
                val telemetry = OnDeviceProcessDriver.executeInference(
                    src, mask, targetModel, backend, posPrompt, negPrompt, smartPrompt = isSmartPrompt
                )
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnRunInpaint.isEnabled = true
                    modelSpinner.isEnabled = true
                    backendSpinner.isEnabled = true
                    resultImageView.setImageBitmap(telemetry.resultBitmap)
                    telemetryCard.visibility = View.VISIBLE

                    val powerStr = if (telemetry.powerWatts != null) "%.2f W".format(telemetry.powerWatts) else "N/A"
                    val energyStr = if (telemetry.energyJoules != null) "%.2f J".format(telemetry.energyJoules) else "N/A"
                    val currentStr = if (telemetry.rawCurrentUa != null) "%d µA (%.1f mA)".format(telemetry.rawCurrentUa, telemetry.rawCurrentUa / 1000.0) else "N/A"
                    val voltageStr = if (telemetry.rawVoltageUv != null) "%.3f V".format(telemetry.rawVoltageUv / 1e6) else "N/A"
                    val deltaSign = if (telemetry.thermalDeltaC >= 0) "+" else ""
                    val deltaStr = "%s%.2f °C".format(deltaSign, telemetry.thermalDeltaC)
                    val tempRangeStr = "%.1f °C ➔ %.1f °C".format(telemetry.startTempC, telemetry.endTempC)

                    telemetryText.text = """
                        Model: ${telemetry.modelName}
                        Backend: ${telemetry.executionMode}
                        Latency: ${telemetry.latencyMs} ms
                        Active Power: $powerStr
                        Active Energy: $energyStr
                        Battery Current: $currentStr
                        Battery Voltage: $voltageStr
                        Peak Temperature: $tempRangeStr
                        Thermal Delta: $deltaStr
                    """.trimIndent()
                    Toast.makeText(this, "✨ Inpainting complete with ${telemetry.modelName}", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnRunInpaint.isEnabled = true
                    modelSpinner.isEnabled = true
                    backendSpinner.isEnabled = true
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
                    val srcPath = i.getStringExtra("src")
                    val src = if (srcPath != null) BitmapFactory.decodeFile(srcPath) else canvasView.getSourceBitmap()
                    val engine = srEngines.firstOrNull { it.id == id }
                    if (src == null || engine == null) "no source or unknown engine $id"
                    else if (engine.unavailableReason != null) "engine disabled: ${engine.unavailableReason}"
                    else {
                        BackendMatrix.require(engine.family, backend)
                        val t0 = System.nanoTime()
                        val out = engine.run(src, backend)
                        val ms = (System.nanoTime() - t0) / 1_000_000
                        i.getStringExtra("out")?.let { p ->
                            File(p).apply { parentFile?.mkdirs() }.outputStream().use { out.compress(Bitmap.CompressFormat.PNG, 100, it) }
                        }
                        Log.i("QIDK.SR", "engine=$id backend=$backend out=${out.width}x${out.height} ms=$ms")
                        "engine=$id backend=$backend out=${out.width}x${out.height} ms=$ms"
                    }
                }
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

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        canvasView.invalidate()
    }
}
