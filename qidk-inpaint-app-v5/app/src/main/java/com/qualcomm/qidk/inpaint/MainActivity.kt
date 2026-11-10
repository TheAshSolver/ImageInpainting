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

class MainActivity : AppCompatActivity() {

    private lateinit var canvasView: InpaintCanvasView
    private lateinit var resultImageView: ImageView
    private lateinit var modelSpinner: Spinner
    private lateinit var runtimeSpinner: Spinner
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
        setContentView(R.layout.activity_main)

        canvasView = findViewById(R.id.inpaintCanvasView)
        resultImageView = findViewById(R.id.resultImageView)
        modelSpinner = findViewById(R.id.modelSpinner)
        runtimeSpinner = findViewById(R.id.runtimeSpinner)
        routerBadgeText = findViewById(R.id.routerBadgeText)
        routerJustificationText = findViewById(R.id.routerJustificationText)
        preloadStatusBadge = findViewById(R.id.preloadStatusBadge)
        promptCard = findViewById(R.id.promptCard)
        etPromptPositive = findViewById(R.id.etPromptPositive)
        etPromptNegative = findViewById(R.id.etPromptNegative)
        telemetryCard = findViewById(R.id.telemetryCard)
        telemetryText = findViewById(R.id.telemetryText)
        progressBar = findViewById(R.id.progressBar)

        setupRuntimeSpinner()
        setupModelSpinner()
        setupButtons()
        triggerModelPreload()
        MediaPipeSegmenter.initialize(applicationContext)

        if (savedInstanceState == null) {
            loadDefaultSample()
        }
    }

    private fun updateModelSpinnerForRuntime(runtime: String) {
        val models = when {
            runtime.contains("GPU") -> arrayOf(
                "LaMa Dilated (Neural GPU)",
                "MIGAN - Faces (Neural GPU)",
                "AOT-GAN (Neural GPU)",
                "Hybrid / Auto"
            )
            runtime.contains("CPU") -> arrayOf(
                "Fast Marching (Telea - Algorithmic CPU)",
                "Fluid Streamlines (Navier-Stokes - Algorithmic CPU)",
                "Criminisi Exemplar (Algorithmic CPU)",
                "Biharmonic Poisson (Algorithmic CPU)",
                "LaMa Dilated (Neural CPU)",
                "MIGAN - Faces (Neural CPU)",
                "AOT-GAN (Neural CPU)",
                "Hybrid / Auto"
            )
            else -> arrayOf(
                "LaMa Dilated (Neural NPU)",
                "MIGAN - Faces (Neural NPU)",
                "AOT-GAN (Neural NPU)",
                "Stable Diffusion 1.5 (Latent Diffusion)",
                "Smart Stable Diffusion (Vision-Prompt Synthesized)",
                "Inpainting 2x Super-Resolution",
                "Hybrid / Auto"
            )
        }

        val previousSelection = modelSpinner.selectedItem?.toString() ?: ""
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, models)
        modelSpinner.adapter = adapter

        // Preserve previous selection if it is feasible on the new hardware runtime
        val matchingIdx = models.indexOfFirst {
            val key = when {
                it.contains("LaMa") -> "LaMa"
                it.contains("MIGAN") -> "MIGAN"
                it.contains("AOT") -> "AOT"
                it.contains("Telea") -> "Telea"
                it.contains("Navier") -> "Navier"
                it.contains("Criminisi") -> "Criminisi"
                it.contains("Biharmonic") -> "Biharmonic"
                it.contains("Diffusion") -> "Diffusion"
                it.contains("Super-Resolution") -> "Super-Resolution"
                it.contains("Hybrid") || it.contains("Auto") -> "Auto"
                else -> ""
            }
            key.isNotEmpty() && previousSelection.contains(key)
        }

        if (matchingIdx >= 0) {
            modelSpinner.setSelection(matchingIdx)
        } else {
            modelSpinner.setSelection(0)
        }

        val selected = modelSpinner.selectedItem?.toString() ?: ""
        promptCard.visibility = if (selected.contains("Diffusion")) View.VISIBLE else View.GONE
    }

    private fun setupModelSpinner() {
        val currentRuntime = runtimeSpinner.selectedItem?.toString() ?: "Hexagon NPU (HTP v79)"
        updateModelSpinnerForRuntime(currentRuntime)

        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = parent?.getItemAtPosition(position)?.toString() ?: ""
                val isSd = selected.contains("Stable Diffusion") || selected.contains("Diffusion")
                promptCard.visibility = if (isSd) View.VISIBLE else View.GONE
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                promptCard.visibility = View.GONE
            }
        }
    }

    private fun setupRuntimeSpinner() {
        val runtimes = arrayOf(
            "Hexagon NPU (HTP v79)",
            "Adreno GPU (OpenCL)",
            "Oryon CPU (NEON)"
        )
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, runtimes)
        runtimeSpinner.adapter = adapter

        runtimeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selectedRuntime = parent?.getItemAtPosition(position)?.toString() ?: "Hexagon NPU"
                updateModelSpinnerForRuntime(selectedRuntime)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
            }
        }
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
        val btnModeCircle = findViewById<Button>(R.id.btnModeCircle)
        val btnModeBrush = findViewById<Button>(R.id.btnModeBrush)
        val btnModeBox = findViewById<Button>(R.id.btnModeBox)
        val btnSegGrabCut = findViewById<Button>(R.id.btnSegGrabCut)
        val btnSegAiSam = findViewById<Button>(R.id.btnSegAiSam)
        val btnSegExact = findViewById<Button>(R.id.btnSegExact)

        fun updateModeButtons(activeMode: InpaintCanvasView.Mode) {
            btnModeCircle?.backgroundTintList = ColorStateList.valueOf(
                if (activeMode == InpaintCanvasView.Mode.CIRCLE_SELECT) Color.parseColor("#7C3AED") else Color.parseColor("#4B5563")
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

        updateModeButtons(InpaintCanvasView.Mode.CIRCLE_SELECT)
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

        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            takePhotoLauncher.launch(null)
        }

        btnModeCircle?.setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.CIRCLE_SELECT)
            updateModeButtons(InpaintCanvasView.Mode.CIRCLE_SELECT)
            Toast.makeText(this, "⭕ Circle Select Mode: Circle any object to mask it", Toast.LENGTH_SHORT).show()
        }

        btnModeBrush?.setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.BRUSH)
            updateModeButtons(InpaintCanvasView.Mode.BRUSH)
            Toast.makeText(this, "🖌️ Brush Mode: Finger draw mask", Toast.LENGTH_SHORT).show()
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
        runtimeSpinner.isEnabled = false
        progressBar.visibility = View.VISIBLE

        val selectedOption = modelSpinner.selectedItem?.toString() ?: "Auto"
        val selectedRuntime = runtimeSpinner.selectedItem?.toString() ?: "Hexagon NPU"

        val targetRuntime = when {
            selectedRuntime.contains("GPU") -> "GPU"
            selectedRuntime.contains("CPU") -> "CPU"
            else -> "NPU"
        }

        val targetModel = when {
            selectedOption.contains("Telea") || selectedOption.contains("Fast Marching") -> "TELEA"
            selectedOption.contains("Navier") || selectedOption.contains("Streamlines") -> "NAVIER_STOKES"
            selectedOption.contains("Hybrid") || selectedOption.contains("Auto") -> {
                val dec = RouterClassifier.classifyAndRoute(src, mask)
                dec.recommendedModel
            }
            selectedOption.contains("MIGAN") -> "MIGAN"
            selectedOption.contains("LaMa") -> "LAMA"
            selectedOption.contains("AOT") -> "AOTGAN"
            selectedOption.contains("Stable Diffusion") || selectedOption.contains("Diffusion") -> "SD"
            else -> "LAMA"
        }

        val posPrompt = etPromptPositive.text.toString().trim()
        val negPrompt = etPromptNegative.text.toString().trim()

        Thread {
            try {
                val telemetry = OnDeviceProcessDriver.executeInference(src, mask, targetModel, posPrompt, negPrompt, targetRuntime)
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnRunInpaint.isEnabled = true
                    modelSpinner.isEnabled = true
                    runtimeSpinner.isEnabled = true
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
                    AlertDialog.Builder(this)
                        .setTitle("❌ Inference Error")
                        .setMessage(e.message ?: "An unexpected error occurred during inference")
                        .setPositiveButton("OK", null)
                        .show()
                }
            }
        }.start()
    }

    override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {
        super.onConfigurationChanged(newConfig)
        canvasView.invalidate()
    }
}
