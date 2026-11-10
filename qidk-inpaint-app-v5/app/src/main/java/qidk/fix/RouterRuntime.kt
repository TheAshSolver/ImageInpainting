package qidk.fix

import android.content.Context
import android.os.BatteryManager
import android.util.Log

/**
 * Process-wide holder of the LinUCB router: loads assets/production_ccb_router.json once, starts the telemetry poller and
 * exposes [policy]. [policy] stays null when the file is missing, cannot be parsed, or needs features this build cannot
 * compute; the reason is logged under QIDK.Router and RouterClassifier then keeps using its rule ladder.
 *
 * Call RouterRuntime.init(applicationContext) once from MainActivity.onCreate.
 */
object RouterRuntime {
    private const val TAG = "QIDK.Router"
    const val ASSET = "production_ccb_router.json"

    @Volatile var policy: RouterPolicy? = null; private set
    @Volatile var poller: TelemetryPoller? = null; private set

    @Synchronized fun init(context: Context) {
        if (poller != null) return
        val bm = context.getSystemService(BatteryManager::class.java)
        val p = TelemetryPoller(batteryPct = { bm?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)?.toFloat() ?: 0f })
        p.start()
        poller = p
        policy = try {
            val text = context.assets.open(ASSET).bufferedReader().use { it.readText() }
            val model = LinUcbModel.parse(text)
            Log.i(TAG, "loaded ${model.schemaNotes}")
            RouterPolicy(model, { p.latest }).also {
                if (it.usable) Log.i(TAG, "policy ready: modes=${it.modes.map { m -> m?.label }} features=${model.featureNames}")
                else Log.w(TAG, "unusable: ${it.problems}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "unusable: cannot load $ASSET: ${e.message}")
            null
        }
    }
}
