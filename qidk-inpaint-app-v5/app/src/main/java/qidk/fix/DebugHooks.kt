package qidk.fix

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.SystemClock
import android.util.Log
import android.view.MotionEvent
import android.view.View

/**
 * adb-driven test hooks, registered ONLY in debuggable builds. Each handler returns one line that is logged under
 * "QIDK.Hook" together with the action, so a script can wait for it:
 *
 *   adb -s 8f27557f shell am broadcast -a qidk.debug.STROKE --es mode smart --es pts "120,300;130,290;..."
 *   adb -s 8f27557f logcat -d -s QIDK.Hook:I
 *
 * The activity registers handlers for the actions it supports (STROKE, CLASSICAL, SR, ROUTE, SD) in onCreate:
 *
 *   DebugHooks.install(this, mapOf("qidk.debug.STROKE" to { e -> ... "ok" }))
 */
object DebugHooks {
    private const val TAG = "QIDK.Hook"

    fun install(context: Context, handlers: Map<String, (Intent) -> String>): BroadcastReceiver? {
        if (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return null
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(c: Context, i: Intent) {
                val action = i.action ?: return
                val t0 = System.nanoTime()
                val line = try { handlers[action]?.invoke(i) ?: "no handler" } catch (t: Throwable) { "ERROR ${t.javaClass.simpleName}: ${t.message}" }
                Log.i(TAG, "$action -> $line (${(System.nanoTime() - t0) / 1_000_000} ms)")
            }
        }
        val filter = IntentFilter().apply { handlers.keys.forEach { addAction(it) } }
        if (Build.VERSION.SDK_INT >= 33) context.registerReceiver(receiver, filter, Context.RECEIVER_EXPORTED)
        else @Suppress("UnspecifiedRegisterReceiverFlag") context.registerReceiver(receiver, filter)
        Log.i(TAG, "installed for ${handlers.keys}")
        return receiver
    }

    /**
     * Runs [work] on a worker thread and logs "<action> -> <result>" when it finishes. Use it for anything that can take
     * more than a few milliseconds: broadcast receivers run on the UI thread and are killed as ANRs after ~10 s.
     */
    fun async(action: String, work: () -> String): String {
        Thread({
            val line = try { work() } catch (t: Throwable) { "ERROR ${t.javaClass.simpleName}: ${t.message}" }
            Log.i(TAG, "$action -> $line")
        }, "hook-$action").start()
        return "started"
    }

    /** "x,y;x,y;..." -> list of (x, y). */
    fun parsePoints(text: String?): List<Pair<Float, Float>> =
        text.orEmpty().split(';').mapNotNull { p ->
            val xy = p.split(',')
            if (xy.size == 2) (xy[0].trim().toFloatOrNull() ?: return@mapNotNull null) to (xy[1].trim().toFloatOrNull() ?: return@mapNotNull null) else null
        }

    /**
     * Plays a finger stroke into [view] (coordinates in the VIEW's own space): DOWN, MOVE..., UP, dispatched on the UI
     * thread through the normal touch path (dispatchTouchEvent), so zoom/pan/mode handling is exercised exactly as for a real finger.
     */
    fun injectStroke(view: View, pts: List<Pair<Float, Float>>, stepMs: Long = 8L): String {
        if (pts.size < 2) return "need at least 2 points"
        view.post {
            val t0 = SystemClock.uptimeMillis()
            for ((i, p) in pts.withIndex()) {
                val action = when (i) { 0 -> MotionEvent.ACTION_DOWN; pts.size - 1 -> MotionEvent.ACTION_UP; else -> MotionEvent.ACTION_MOVE }
                val e = MotionEvent.obtain(t0, t0 + i * stepMs, action, p.first, p.second, 0)
                view.dispatchTouchEvent(e)
                e.recycle()
            }
        }
        return "injected ${pts.size} points"
    }
}
