package com.qualcomm.qidk.inpaint

import android.app.Activity
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
import com.qualcomm.qidk.inpaint.engine.DeepMaskRefiner
import com.qualcomm.qidk.inpaint.engine.MediaPipeSegmenter
import com.qualcomm.qidk.inpaint.engine.ModelPreloadManager
import com.qualcomm.qidk.inpaint.engine.OnDeviceProcessDriver
import com.qualcomm.qidk.inpaint.router.RouterClassifier
import com.qualcomm.qidk.inpaint.ui.InpaintCanvasView
import com.qualcomm.qidk.inpaint.utils.GrabCutEngine
import java.io.File
import java.io.InputStream
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var canvasView: InpaintCanvasView
    private lateinit var resultImageView: ImageView
    private lateinit var modelSpinner: Spinner
    private lateinit var routerBadgeText: TextView
    private lateinit var routerJustificationText: TextView
    private lateinit var telemetryCard: View
    private lateinit var telemetryText: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var promptContainer: View
    private lateinit var editPrompt: EditText

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
        telemetryCard = findViewById(R.id.telemetryCard)
        telemetryText = findViewById(R.id.telemetryText)
        progressBar = findViewById(R.id.progressBar)
        promptContainer = findViewById(R.id.promptContainer)
        editPrompt = findViewById(R.id.editPrompt)

        setupModelSpinner()
        setupButtons()
        MediaPipeSegmenter.initialize(applicationContext)
        ModelPreloadManager.preloadAndWarmup(applicationContext) { success, statusText ->
            runOnUiThread {
                if (success) {
                    Toast.makeText(this, "⚡ Models preloaded in RAM & NPU hot", Toast.LENGTH_SHORT).show()
                }
            }
        }

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
            "Stable Diffusion 1.5 (Inefficient RePaint - Euler 20-step)"
        )
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, models)
        modelSpinner.adapter = adapter

        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = modelSpinner.selectedItem?.toString() ?: ""
                val isSd = selected.contains("Stable Diffusion") || selected.contains("SD")
                promptContainer.visibility = if (isSd) View.VISIBLE else View.GONE
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
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
        }

        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            takePhotoLauncher.launch(null)
        }

        val btnModeBrush = findViewById<Button>(R.id.btnModeBrush)
        val btnModeBox = findViewById<Button>(R.id.btnModeBox)

        fun updateModeButtons() {
            if (canvasView.getMode() == InpaintCanvasView.Mode.BRUSH) {
                btnModeBrush.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#2563EB"))
                btnModeBox.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#4B5563"))
            } else {
                btnModeBrush.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#4B5563"))
                btnModeBox.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#2563EB"))
            }
        }
        updateModeButtons()

        btnModeBrush.setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.BRUSH)
            updateModeButtons()
            Toast.makeText(this, "🖌️ Brush Mode: Finger paint mask", Toast.LENGTH_SHORT).show()
        }

        btnModeBox.setOnClickListener {
            canvasView.setMode(InpaintCanvasView.Mode.BOUNDING_BOX)
            updateModeButtons()
            Toast.makeText(this, "🔲 Box Select: Tap 2 corners or drag to select everything inside box", Toast.LENGTH_SHORT).show()
        }

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            canvasView.clearMask()
            updateRouterLogic()
        }

        findViewById<Button>(R.id.btnUndo).setOnClickListener {
            canvasView.undo()
            updateRouterLogic()
        }

        findViewById<Button>(R.id.btnZoomIn).setOnClickListener {
            canvasView.zoomIn()
        }

        findViewById<Button>(R.id.btnZoomOut).setOnClickListener {
            canvasView.zoomOut()
        }

        findViewById<Button>(R.id.btnZoomReset).setOnClickListener {
            canvasView.resetZoom()
        }

        findViewById<Button>(R.id.btnLoadSample)?.setOnClickListener {
            showSamplePickerDialog()
        }

        findViewById<Button>(R.id.btnRunInpaint).setOnClickListener {
            runInpainting()
        }
    }

    private fun showSamplePickerDialog() {
        // Direct sample loading on Android 15 (Scoped Storage compliant)
        val internalDir = File(filesDir, "samples")
        val samplesDir = File(getExternalFilesDir(null), "samples")
        val altDir = File("/sdcard/Android/data/com.qualcomm.qidk.inpaint/files/samples")
        val picDir = File("/sdcard/Pictures/Inpainting102")
        val targetDir = when {
            internalDir.exists() && internalDir.isDirectory && (internalDir.listFiles()?.isNotEmpty() == true) -> internalDir
            samplesDir.exists() && samplesDir.isDirectory && (samplesDir.listFiles()?.isNotEmpty() == true) -> samplesDir
            altDir.exists() && altDir.isDirectory && (altDir.listFiles()?.isNotEmpty() == true) -> altDir
            picDir.exists() && picDir.isDirectory && (picDir.listFiles()?.isNotEmpty() == true) -> picDir
            internalDir.exists() && internalDir.isDirectory -> internalDir
            samplesDir.exists() && samplesDir.isDirectory -> samplesDir
            altDir.exists() && altDir.isDirectory -> altDir
            picDir.exists() && picDir.isDirectory -> picDir
            else -> null
        }

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
        val picDir = File("/sdcard/Pictures/Inpainting102")
        val targetDir = when {
            internalDir.exists() && internalDir.isDirectory && File(internalDir, "001.png").exists() -> internalDir
            samplesDir.exists() && samplesDir.isDirectory && File(samplesDir, "001.png").exists() -> samplesDir
            altDir.exists() && altDir.isDirectory && File(altDir, "001.png").exists() -> altDir
            picDir.exists() && picDir.isDirectory && File(picDir, "001.png").exists() -> picDir
            internalDir.exists() && internalDir.isDirectory -> internalDir
            samplesDir.exists() && samplesDir.isDirectory -> samplesDir
            altDir.exists() && altDir.isDirectory -> altDir
            picDir.exists() && picDir.isDirectory -> picDir
            else -> null
        }

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

    private fun updateRouterLogic(autoSelectModel: Boolean = false) {
        val src = canvasView.getSourceBitmap() ?: return
        val mask = canvasView.getMaskBitmap()

        val decision = RouterClassifier.classifyAndRoute(src, mask)

        val color = when (decision.recommendedModel) {
            "MIGAN" -> Color.parseColor("#10B981")
            "AOTGAN" -> Color.parseColor("#3B82F6")
            else -> Color.parseColor("#8B5CF6")
        }

        routerBadgeText.setTextColor(color)
        routerBadgeText.text = "🎯 AUTO-SELECTED: ${decision.recommendedModel} (${decision.ruleTriggered})"
        routerJustificationText.text = decision.justification

        if (autoSelectModel) {
            val targetIdx = when (decision.recommendedModel) {
                "MIGAN" -> 1
                "LAMA" -> 2
                "AOTGAN" -> 3
                else -> 1
            }
            modelSpinner.setSelection(targetIdx)
        }
    }

    private fun runInpainting() {
        val src = canvasView.getSourceBitmap()
        val mask = canvasView.getMaskBitmap()

        if (src == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
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
            selectedOption.contains("Inefficient") -> "SD_INEFFICIENT"
            selectedOption.contains("Inpainting") || selectedOption.contains("Diffusion") || selectedOption.contains("SD") -> "SD"
            else -> "MIGAN"
        }

        val customPrompt = editPrompt.text?.toString()?.trim() ?: ""

        Thread {
            try {
                val telemetry = OnDeviceProcessDriver.executeInference(src, mask, targetModel, customPrompt)
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    btnRunInpaint.isEnabled = true
                    modelSpinner.isEnabled = true
                    resultImageView.setImageBitmap(telemetry.resultBitmap)
                    telemetryCard.visibility = View.VISIBLE
                    val thermalText = if (telemetry.peakTempC > 0f) {
                        "+${String.format(Locale.US, "%.1f", telemetry.thermalDeltaC)} °C (Peak: ${String.format(Locale.US, "%.1f", telemetry.peakTempC)} °C)"
                    } else {
                        "+${String.format(Locale.US, "%.1f", telemetry.thermalDeltaC)} °C"
                    }
                    val ramText = if (telemetry.usedRamGb > 0f) {
                        "${String.format(Locale.US, "%.2f", telemetry.usedRamGb)} GB"
                    } else {
                        "N/A"
                    }
                    telemetryText.text = """
                        Status: ${telemetry.executionMode}
                        Model Executed: ${telemetry.modelName}
                        Latency: ${telemetry.latencyMs} ms
                        Active Energy: ${telemetry.energyJoules} Joules
                        Active Power: ${telemetry.powerWatts} Watts
                        Thermal Delta: $thermalText
                        Active RAM: $ramText
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
