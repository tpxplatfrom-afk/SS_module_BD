package ai.nano.engine

/**
 * Typed Exception hierarchy representing native THSA-2B engine error conditions.
 */
open class NanoEngineException(
    message: String,
    val errorCode: Int
) : RuntimeException("NanoEngine Error [$errorCode]: $message")

class NanoOomException(message: String) : NanoEngineException(message, -2)
class NanoCancelledException(message: String) : NanoEngineException(message, -3)
class NanoCorruptModelException(message: String) : NanoEngineException(message, -4)
class NanoInvalidTokenException(message: String) : NanoEngineException(message, -5)
class NanoBusyException(message: String) : NanoEngineException(message, -6)
