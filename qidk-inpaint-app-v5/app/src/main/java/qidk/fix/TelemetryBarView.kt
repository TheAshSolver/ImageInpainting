package qidk.fix

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView
import com.qualcomm.qidk.inpaint.engine.OnDeviceProcessDriver
import java.io.File
import java.util.Locale

/**
 * Reusable Live Hardware Telemetry HUD Bar.
 * Updates every 500ms directly from sysfs and TelemetryPoller:
 * - Silicon Junction Temperature (°C) with color-coded safety levels.
 * - Active Instantaneous Power (W) and Current (mA).
 * - Hardware Core Contention (CPU %, GPU %, NPU status).
 * - Thermal Governor Status (Normal vs Throttled).
 */
class TelemetryBarView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : LinearLayout(context, attrs, defStyleAttr) {

    private val tempText = TextView(context)
    private val powerText = TextView(context)
    private val loadText = TextView(context)
    private val govText = TextView(context)

    private val handler = Handler(Looper.getMainLooper())
    private var isPolling = false

    private val pollRunnable = object : Runnable {
        override fun run() {
            updateTelemetry()
            if (isPolling) {
                handler.postDelayed(this, 500L)
            }
        }
    }

    init {
        orientation = HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        val density = resources.displayMetrics.density
        val padH = (10 * density).toInt()
        val padV = (6 * density).toInt()
        setPadding(padH, padV, padH, padV)

        val gd = GradientDrawable().apply {
            setColor(Color.parseColor("#161F30"))
            cornerRadius = 10 * density
            setStroke((1 * density).toInt(), Color.parseColor("#1E293B"))
        }
        background = gd

        fun setupBadge(tv: TextView, weight: Float): TextView {
            tv.typeface = Typeface.MONOSPACE
            tv.textSize = 11f
            tv.gravity = Gravity.CENTER
            tv.setTextColor(Color.parseColor("#F8FAFC"))
            val lp = LayoutParams(0, LayoutParams.WRAP_CONTENT, weight)
            addView(tv, lp)
            return tv
        }

        setupBadge(tempText, 1.1f)
        setupBadge(powerText, 1.3f)
        setupBadge(loadText, 1.4f)
        setupBadge(govText, 1.0f)

        updateTelemetry()
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        isPolling = true
        handler.post(pollRunnable)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        isPolling = false
        handler.removeCallbacks(pollRunnable)
    }

    fun setThemeColors(surfaceColor: Int, borderColor: Int, textColor: Int) {
        val density = resources.displayMetrics.density
        val gd = GradientDrawable().apply {
            setColor(surfaceColor)
            cornerRadius = 10 * density
            setStroke((1 * density).toInt(), borderColor)
        }
        background = gd
        powerText.setTextColor(textColor)
        loadText.setTextColor(textColor)
    }

    private fun updateTelemetry() {
        // 1. Temperature & Color-Coding
        val tempC = OnDeviceProcessDriver.readPeakTemperatureC()
        tempText.text = String.format(Locale.US, "🌡️ %.1f°C", tempC)
        when {
            tempC >= 70.0f -> tempText.setTextColor(Color.parseColor("#EF4444")) // Red
            tempC >= 55.0f -> tempText.setTextColor(Color.parseColor("#F59E0B")) // Amber
            else -> tempText.setTextColor(Color.parseColor("#10B981"))           // Emerald Green
        }

        // 2. Power & Current
        val powerWatts = OnDeviceProcessDriver.sampleInstantaneousPowerWatts()
        val currentUa = OnDeviceProcessDriver.readActiveCurrentUa()
        if (powerWatts != null && powerWatts > 0f) {
            val currentMa = if (currentUa != null) currentUa / 1000L else 0L
            powerText.text = String.format(Locale.US, "⚡ %.1fW (%dmA)", powerWatts, currentMa)
        } else if (currentUa != null && currentUa != 0L) {
            powerText.text = String.format(Locale.US, "⚡ %dmA", currentUa / 1000L)
        } else {
            powerText.text = "⚡ Active"
        }

        // 3. Hardware Load (CPU/GPU)
        val poller = RouterRuntime.poller
        val cpuPct = if (poller != null) (poller.latest.cpuBusy * 100f).toInt() else 0
        val gpuPct = if (poller != null) (poller.latest.gpuBusy * 100f).toInt() else 0
        loadText.text = "CPU:${cpuPct}%|GPU:${gpuPct}%"

        // 4. Governor & Thermal Throttling Status
        val isThrottled = checkThermalThrottling(tempC)
        if (isThrottled) {
            govText.text = "Gov:Throttled"
            govText.setTextColor(Color.parseColor("#EF4444"))
        } else {
            govText.text = "Gov:Normal"
            govText.setTextColor(Color.parseColor("#38BDF8"))
        }
    }

    private fun checkThermalThrottling(tempC: Float): Boolean {
        if (tempC >= 74.0f) return true
        return try {
            val cd38 = File("/sys/class/thermal/cooling_device38/cur_state")
            if (cd38.exists() && cd38.canRead()) {
                val state = cd38.readText().trim().toIntOrNull() ?: 0
                state > 0
            } else false
        } catch (_: Exception) {
            false
        }
    }
}
