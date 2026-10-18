package com.qualcomm.qidk.inpaint

import android.app.Activity
import android.content.Intent
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
import com.qualcomm.qidk.inpaint.ui.InpaintCanvasView
import com.qualcomm.qidk.inpaint.utils.GrabCutEngine
import java.io.File
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var canvasView: InpaintCanvasView
    private lateinit var resultImageView: ImageView
    private lateinit var modelSpinner: Spinner
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

        if (savedInstanceState == null) {
            loadDefaultSample()
        }
    }

    private fun setupModelSpinner() {
        val models = arrayOf(
            "Auto (Router Recommended)",
            "MIGAN (Portraits & Speed)",
            "LaMa Dilated (Large Voids)",
            "AOT-GAN (Dense Texture)",
            "Stable Diffusion 1.5 (Inpainting - DPM 12-step)",
            "Stable Diffusion 1.5 (New Inpaint - 20-step HD)",
            "Stable Diffusion 1.5 (Inefficient RePaint - Euler 20-step)"
        )
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, models)
        modelSpinner.adapter = adapter

        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = parent?.getItemAtPosition(position)?.toString() ?: ""
                val isSd = selected.contains("Stable Diffusion") || selected.contains("SD")
                promptCard.visibility = if (isSd) View.VISIBLE else View.GONE
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                promptCard.visibility = View.GONE
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
        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        canvasView.onBoxSelectionStatusChanged = { msg ->
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
        }

        canvasView.onMaskChanged = {
            updateRouterLogic()
            val src = canvasView.getSourceBitmap()
            val mask = canvasView.getMaskBitmap()
            if (src != null) {
                val selectedOption = modelSpinner.selectedItem?.toString() ?: "Auto"
                val isMigan = if (selectedOption.contains("Auto")) {
                    val dec = RouterClassifier.classifyAndRoute(src, mask)
                    dec.recommendedModel == "MIGAN"
                } else {
                    selectedOption.contains("MIGAN")
                }
                OnDeviceProcessDriver.eagerPrecacheMask(mask, inverted = isMigan)
            }
        }

        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            takePhotoLauncher.launch(null)
        }

        findViewById<Button>(R.id.btnModeBrush).setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.BRUSH)
            Toast.makeText(this, "Brush Mode: Finger draw mask", Toast.LENGTH_SHORT).show()
        }

        findViewById<Button>(R.id.btnModeBox).setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.BOUNDING_BOX)
            Toast.makeText(this, "Box Mode: Tap 1st corner, then opposite corner", Toast.LENGTH_SHORT).show()
        }

        findViewById<Button>(R.id.btnGrabCut).setOnClickListener {
            val box = canvasView.getSelectionBox()
            val src = canvasView.getSourceBitmap()
            if (box != null && src != null) {
                val (mask, latency) = GrabCutEngine.generateBoundingBoxMask(src, box)
                canvasView.applyGrabCutMask(mask)
                canvasView.resetBoxSelection()
                updateRouterLogic()
                Toast.makeText(this, "GrabCut generated silhouette in ${latency}ms", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Please tap 2 corners to select a box first", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<Button>(R.id.btnRefine).setOnClickListener {
            val src = canvasView.getSourceBitmap()
            val roughMask = canvasView.getMaskBitmap()
            if (src != null && roughMask != null) {
                val (refined, latency) = GrabCutEngine.refineMaskSnapToEdges(src, roughMask)
                canvasView.applyGrabCutMask(refined)
                updateRouterLogic()
                Toast.makeText(this, "✨ Mask snapped to edges in ${latency}ms", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Please paint a rough mask first", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            canvasView.clearMask()
            updateRouterLogic()
        }

        findViewById<Button>(R.id.btnUndo).setOnClickListener {
            canvasView.undo()
            updateRouterLogic()
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
        val v3Dir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint.v3/files/samples")
        val v2Dir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint.v2/files/samples")
        val picDir = File("/sdcard/Pictures/Inpainting102")
        val picDirAlt = File("/sdcard/Pictures/Inpainting")
        val lamaDir = File("/data/local/tmp/lama/samples")
        val candidates = listOf(internalDir, samplesDir, altDir, v3Dir, v2Dir, picDir, picDirAlt, lamaDir)
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
        val v3Dir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint.v3/files/samples")
        val v2Dir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint.v2/files/samples")
        val picDir = File("/sdcard/Pictures/Inpainting102")
        val picDirAlt = File("/sdcard/Pictures/Inpainting")
        val lamaDir = File("/data/local/tmp/lama/samples")
        val candidates = listOf(internalDir, samplesDir, altDir, v3Dir, v2Dir, picDir, picDirAlt, lamaDir)
        val targetDir = candidates.firstOrNull { it.exists() && it.isDirectory && File(it, "001.png").exists() }
            ?: candidates.firstOrNull { it.exists() && it.isDirectory && (it.listFiles()?.isNotEmpty() == true) }

        if (targetDir != null) {
            val s001 = File(targetDir, "001.png")
            if (s001.exists()) {
                loadSampleFile(s001, targetDir)
                return
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
        // V3 Eager Pre-caching: Serialize image to live_img_v3.raw in background immediately
        OnDeviceProcessDriver.eagerPrecacheImage(scaled)
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
        progressBar.visibility = View.VISIBLE

        val selectedOption = modelSpinner.selectedItem?.toString() ?: "Auto"

        val targetModel = when {
            selectedOption.contains("Auto") -> {
                val dec = RouterClassifier.classifyAndRoute(src, mask)
                dec.recommendedModel
            }
            selectedOption.contains("MIGAN") -> "MIGAN"
            selectedOption.contains("LaMa") -> "LAMA"
            selectedOption.contains("AOT") -> "AOTGAN"
            selectedOption.contains("New Inpaint") || selectedOption.contains("HD") -> "SD_NEW_INPAINT"
            selectedOption.contains("Inefficient") -> "SD_INEFFICIENT"
            selectedOption.contains("Inpainting") || selectedOption.contains("Diffusion") || selectedOption.contains("SD") -> "SD"
            else -> "MIGAN"
        }

        val posPrompt = etPromptPositive.text.toString().trim()
        val negPrompt = etPromptNegative.text.toString().trim()

        val clickTimestamp = android.os.SystemClock.elapsedRealtime()

        Thread {
            try {
                val telemetry = OnDeviceProcessDriver.executeInference(src, mask, targetModel, posPrompt, negPrompt)
                runOnUiThread {
                    val clickToScreenMs = android.os.SystemClock.elapsedRealtime() - clickTimestamp
                    progressBar.visibility = View.GONE
                    btnRunInpaint.isEnabled = true
                    modelSpinner.isEnabled = true
                    resultImageView.setImageBitmap(telemetry.resultBitmap)
                    telemetryCard.visibility = View.VISIBLE

                    val powerStr = if (telemetry.powerWatts != null) "%.2f W".format(telemetry.powerWatts) else "N/A"
                    val energyStr = if (telemetry.energyJoules != null) "%.2f J".format(telemetry.energyJoules) else "N/A"
                    val currentStr = if (telemetry.rawCurrentUa != null) "%d µA (%.1f mA)".format(telemetry.rawCurrentUa, telemetry.rawCurrentUa / 1000.0) else "N/A"
                    val voltageStr = if (telemetry.rawVoltageUv != null) "%.3f V".format(telemetry.rawVoltageUv / 1e6) else "N/A"
                    val deltaSign = if (telemetry.thermalDeltaC >= 0) "+" else ""
                    val deltaStr = "%s%.2f °C".format(deltaSign, telemetry.thermalDeltaC)
                    val tempRangeStr = "%.1f °C ➔ %.1f °C".format(telemetry.startTempC, telemetry.endTempC)
                    val stagingStr = if (telemetry.stagingLatencyMs <= 2) "0 ms (Pre-cached ⚡)" else "${telemetry.stagingLatencyMs} ms"

                    telemetryText.text = """
                        Model: ${telemetry.modelName}
                        Backend: ${telemetry.executionMode}
                        Click-to-Screen Latency: ${clickToScreenMs} ms
                        NPU Engine Execution: ${telemetry.engineLatencyMs} ms
                        Input Staging Overhead: $stagingStr
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
