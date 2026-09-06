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
import androidx.appcompat.app.AppCompatActivity
import com.qualcomm.qidk.inpaint.engine.OnDeviceProcessDriver
import com.qualcomm.qidk.inpaint.router.RouterClassifier
import com.qualcomm.qidk.inpaint.ui.InpaintCanvasView
import com.qualcomm.qidk.inpaint.utils.GrabCutEngine
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var canvasView: InpaintCanvasView
    private lateinit var resultImageView: ImageView
    private lateinit var modelSpinner: Spinner
    private lateinit var routerBadgeText: TextView
    private lateinit var routerJustificationText: TextView
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
        telemetryCard = findViewById(R.id.telemetryCard)
        telemetryText = findViewById(R.id.telemetryText)
        progressBar = findViewById(R.id.progressBar)

        setupModelSpinner()
        setupButtons()
        loadDefaultSample()
    }

    private fun setupModelSpinner() {
        val models = arrayOf(
            "Auto (Router Recommended)",
            "MIGAN (Portraits & Speed)",
            "LaMa Dilated (Large Voids)",
            "AOT-GAN (Dense Texture)",
            "Stable Diffusion 1.5 (RePaint)"
        )
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, models)
        modelSpinner.adapter = adapter
    }

    private fun setupButtons() {
        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            pickImageLauncher.launch("image/*")
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
            Toast.makeText(this, "Box Mode: Drag box around unwanted object", Toast.LENGTH_SHORT).show()
        }

        findViewById<Button>(R.id.btnGrabCut).setOnClickListener {
            val box = canvasView.getSelectionBox()
            val src = canvasView.getSourceBitmap()
            if (box != null && src != null) {
                val (mask, latency) = GrabCutEngine.generateBoundingBoxMask(src, box)
                canvasView.applyGrabCutMask(mask)
                updateRouterLogic()
                Toast.makeText(this, "GrabCut generated silhouette in ${latency}ms", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Please drag a selection box first", Toast.LENGTH_SHORT).show()
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

        findViewById<Button>(R.id.btnRunInpaint).setOnClickListener {
            runInpainting()
        }
    }

    private fun loadDefaultSample() {
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

        progressBar.visibility = View.VISIBLE
        val selectedOption = modelSpinner.selectedItem.toString()

        val targetModel = when {
            selectedOption.contains("Auto") -> {
                val dec = RouterClassifier.classifyAndRoute(src, mask)
                dec.recommendedModel
            }
            selectedOption.contains("MIGAN") -> "MIGAN"
            selectedOption.contains("LaMa") -> "LAMA"
            selectedOption.contains("AOT") -> "AOTGAN"
            selectedOption.contains("Diffusion") -> "SD"
            else -> "MIGAN"
        }

        Thread {
            val telemetry = OnDeviceProcessDriver.executeInference(src, mask, targetModel)
            runOnUiThread {
                progressBar.visibility = View.GONE
                resultImageView.setImageBitmap(telemetry.resultBitmap)
                telemetryCard.visibility = View.VISIBLE
                telemetryText.text = """
                    Status: ${telemetry.executionMode}
                    Model Executed: ${telemetry.modelName}
                    Latency: ${telemetry.latencyMs} ms
                    Active Energy: ${telemetry.energyJoules} Joules
                    Active Power: ${telemetry.powerWatts} Watts
                    Thermal Delta: +${telemetry.thermalDeltaC} °C
                """.trimIndent()
            }
        }.start()
    }
}
