package qidk.fix

/**
 * Tiny JSON reader (no android.* / org.json dependency, so it runs in plain JVM unit tests).
 * Objects -> LinkedHashMap<String, Any?>, arrays -> ArrayList<Any?>, numbers -> Double, plus Python's NaN / Infinity.
 */
object MiniJson {
    fun parse(text: String): Any? {
        val p = Parser(text)
        p.skipWs()
        val v = p.value()
        p.skipWs()
        require(p.eof()) { "trailing characters at offset ${p.i}" }
        return v
    }

    private class Parser(val s: String) {
        var i = 0
        fun eof() = i >= s.length
        fun skipWs() { while (i < s.length && s[i].isWhitespace()) i++ }
        private fun fail(msg: String): Nothing = throw IllegalArgumentException("JSON: $msg at offset $i")
        private fun expect(c: Char) { if (i >= s.length || s[i] != c) fail("expected '$c'"); i++ }
        private fun lit(word: String): Boolean {
            if (s.startsWith(word, i)) { i += word.length; return true }
            return false
        }

        fun value(): Any? {
            skipWs()
            if (eof()) fail("unexpected end")
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> if (lit("true")) true else fail("bad literal")
                'f' -> if (lit("false")) false else fail("bad literal")
                'n' -> if (lit("null")) null else fail("bad literal")
                'N' -> if (lit("NaN")) Double.NaN else fail("bad literal")
                'I' -> if (lit("Infinity")) Double.POSITIVE_INFINITY else fail("bad literal")
                '-' -> if (lit("-Infinity")) Double.NEGATIVE_INFINITY else num()
                else -> if (c == '+' || c == '.' || c.isDigit()) num() else fail("unexpected '$c'")
            }
        }

        private fun obj(): Map<String, Any?> {
            expect('{')
            val m = LinkedHashMap<String, Any?>()
            skipWs()
            if (i < s.length && s[i] == '}') { i++; return m }
            while (true) {
                skipWs()
                val k = str()
                skipWs(); expect(':')
                m[k] = value()
                skipWs()
                if (i < s.length && s[i] == ',') { i++; continue }
                expect('}'); return m
            }
        }

        private fun arr(): List<Any?> {
            expect('[')
            val l = ArrayList<Any?>()
            skipWs()
            if (i < s.length && s[i] == ']') { i++; return l }
            while (true) {
                l.add(value())
                skipWs()
                if (i < s.length && s[i] == ',') { i++; continue }
                expect(']'); return l
            }
        }

        private fun str(): String {
            expect('"')
            val sb = StringBuilder()
            while (true) {
                if (eof()) fail("unterminated string")
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        if (eof()) fail("bad escape")
                        when (val e = s[i++]) {
                            '"', '\\', '/' -> sb.append(e)
                            'b' -> sb.append('\b'); 'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n'); 'r' -> sb.append('\r'); 't' -> sb.append('\t')
                            'u' -> {
                                if (i + 4 > s.length) fail("bad \\u escape")
                                sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4
                            }
                            else -> fail("bad escape \\$e")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun num(): Double {
            val st = i
            while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
            return s.substring(st, i).toDoubleOrNull() ?: fail("bad number '${s.substring(st, i)}'")
        }
    }
}
