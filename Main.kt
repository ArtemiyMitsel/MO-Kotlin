import kotlinx.coroutines.*
import java.awt.Color
import java.awt.Font
import java.awt.image.BufferedImage
import java.io.File
import java.util.*
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

fun quadratic(point: DoubleArray): Double {
    return (point[0] - 2).pow(2) + (point[1] + 3).pow(2)
}

fun quadraticGrad(point: DoubleArray): DoubleArray {
    val dfdx0 = 2 * (point[0] - 2)
    val dfdx1 = 2 * (point[1] + 3)
    return doubleArrayOf(dfdx0, dfdx1)
}

fun rosenbrock(point: DoubleArray): Double {
    return (1 - point[0]).pow(2) + 100 * (point[1] - point[0].pow(2)).pow(2)
}

fun rosenbrockGrad(point: DoubleArray): DoubleArray {
    val dfdx0 = -2 * (1 - point[0]) - 400 * point[0] * (point[1] - point[0].pow(2))
    val dfdx1 = 200 * (point[1] - point[0].pow(2))
    return doubleArrayOf(dfdx0, dfdx1)
}

fun heavyQuadratic(point: DoubleArray): Double {
    return (point[0] - 2).pow(2) + 0.0004 * (point[1] + 3).pow(2)
}

fun heavyQuadraticGrad(point: DoubleArray): DoubleArray {
    val dfDx = 2 * (point[0] - 2)
    val dfDy = 0.0008 * (point[1] + 3)
    return doubleArrayOf(dfDx, dfDy)
}

fun gradientDescentWithTrajectory(
    gradFunction: (DoubleArray) -> DoubleArray,
    startPoint: DoubleArray,
    learningRate: Double = 0.001,
    tolerance: Double = 1e-6,
    maxIterations: Int = 1000000
): Pair<DoubleArray, List<DoubleArray>> {
    var point = startPoint.copyOf()
    val trajectory = mutableListOf(point.copyOf())
    for (i in 0..<maxIterations) {
        val grad = gradFunction(point)
        val pointNew = DoubleArray(point.size) { point[it] - learningRate * grad[it] }
        if (sqrt(pointNew.zip(point) { a, b -> (a - b).pow(2) }.sum()) < tolerance) break
        point = pointNew
        trajectory.add(point.copyOf())
    }
    return Pair(point, trajectory)
}

fun gradientDescentDichotomyWithTrajectory(
    function: (DoubleArray) -> Double,
    gradFunction: (DoubleArray) -> DoubleArray,
    startPoint: DoubleArray,
    tolerance: Double = 1e-6,
    maxIterations: Int = 10000000
): Pair<DoubleArray, List<DoubleArray>> {
    var point = startPoint.copyOf()
    val trajectory = mutableListOf(point.copyOf())
    for (i in 0..<maxIterations) {
        val grad = gradFunction(point)

        val fNew = { alpha: Double -> function(DoubleArray(point.size) { point[it] - alpha * grad[it] }) }

        var alphaLeft = 0.0
        var alphaRight = 1.0
        while (alphaRight - alphaLeft > tolerance) {
            val alphaMid = (alphaLeft + alphaRight) / 2
            if (fNew(alphaMid - tolerance / 2) < fNew(alphaMid + tolerance / 2)) {
                alphaRight = alphaMid
            } else {
                alphaLeft = alphaMid
            }
        }
        val alphaOptimal = (alphaLeft + alphaRight) / 2
        val pointNew = DoubleArray(point.size) { point[it] - alphaOptimal * grad[it] }
        if (sqrt(pointNew.zip(point) { a, b -> (a - b).pow(2) }.sum()) < tolerance) break
        point = pointNew
        trajectory.add(point.copyOf())
    }
    return Pair(point, trajectory)
}

fun gradientDescentGoldenSectionWithTrajectory(
    function: (DoubleArray) -> Double,
    gradFunction: (DoubleArray) -> DoubleArray,
    startPoint: DoubleArray,
    tolerance: Double = 1e-6,
    maxIterations: Int = 100000
): Pair<DoubleArray, List<DoubleArray>> {
    val goldenRatio = (sqrt(5.0) - 1) / 2
    var point = startPoint.copyOf()
    val trajectory = mutableListOf(point.copyOf())

    for (i in 0..<maxIterations) {
        val grad = gradFunction(point)

        val fNew = { alpha: Double -> function(DoubleArray(point.size) { point[it] - alpha * grad[it] }) }

        var a = 0.0
        var b = 1.0
        var c = b - goldenRatio * (b - a)
        var d = a + goldenRatio * (b - a)
        while (abs(c - d) > tolerance) {
            if (fNew(c) < fNew(d)) {
                b = d
            } else {
                a = c
            }
            c = b - goldenRatio * (b - a)
            d = a + goldenRatio * (b - a)
        }
        val alphaOptimal = (b + a) / 2
        val pointNew = DoubleArray(point.size) { point[it] - alphaOptimal * grad[it] }
        if (sqrt(pointNew.zip(point) { a, b -> (a - b).pow(2) }.sum()) < tolerance) break
        point = pointNew
        trajectory.add(point.copyOf())
    }
    return Pair(point, trajectory)
}

fun plotContourAndTrajectory(
    function: (DoubleArray) -> Double,
    startPoint: DoubleArray,
    xlim: Pair<Double, Double>,
    ylim: Pair<Double, Double>,
    titlePrefix: String,
    trajectories: Map<String, List<DoubleArray>>,
    imgName: String? = null
) {
    val width = 800
    val height = 800
    val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val graphics = image.graphics

    graphics.color = Color.WHITE
    graphics.fillRect(0, 0, width, height)

    graphics.color = Color.LIGHT_GRAY
    for (i in 0..10) {
        val x = (i / 10.0 * width).toInt()
        graphics.drawLine(x, 0, x, height)
        val y = (i / 10.0 * height).toInt()
        graphics.drawLine(0, y, width, y)
    }

    val xStep = (xlim.second - xlim.first) / width
    val yStep = (ylim.second - ylim.first) / height
    val zValues = Array(width) { DoubleArray(height) }
    var maxZValue = Double.MIN_VALUE
    var minZValue = Double.MAX_VALUE
    for (i in 0..<width) {
        for (j in 0..<height) {
            val x = xlim.first + i * xStep
            val y = ylim.first + j * yStep
            zValues[i][j] = function(doubleArrayOf(x, y))
            if (zValues[i][j] > maxZValue) {
                maxZValue = zValues[i][j]
            }
            if (zValues[i][j] < minZValue) {
                minZValue = zValues[i][j]
            }
        }
    }

    // неприятная часть - рисование дуг
    val levels = 20
    val deltaZ = (maxZValue - minZValue) / levels
    for (level in 0..levels) {
        val contourValue = minZValue + deltaZ * level
        graphics.color = Color(
            (255 * level / levels).coerceIn(0, 255),
            (255 * (levels - level) / levels).coerceIn(0, 255),
            0
        )
        for (i in 0..<width - 1) {
            for (j in 0..<height - 1) {
                val x1 = xlim.first + i * xStep
                val y1 = ylim.first + j * yStep
                x1 + xStep
                y1 + yStep

                val z1 = zValues[i][j]
                val z2 = zValues[i + 1][j]
                val z3 = zValues[i][j + 1]
                val z4 = zValues[i + 1][j + 1]

                if ((z1 < contourValue && z2 >= contourValue) || (z1 >= contourValue && z2 < contourValue)) {
                    val t = (contourValue - z1) / (z2 - z1)
                    val xc = i + t
                    val yc = j
                    graphics.drawLine(
                        (xc * width / (width - 1)).toInt(), ((height - yc) * height / (height - 1)),
                        ((xc + xStep) * width / (width - 1)).toInt(), ((height - yc) * height / (height - 1))
                    )
                }
                if ((z1 < contourValue && z3 >= contourValue) || (z1 >= contourValue && z3 < contourValue)) {
                    val t = (contourValue - z1) / (z3 - z1)
                    val xc = i
                    val yc = j + t
                    graphics.drawLine(
                        (xc * width / (width - 1)), ((height - yc) * height / (height - 1)).toInt(),
                        (xc * width / (width - 1)), ((height - (yc + yStep)) * height / (height - 1)).toInt()
                    )
                }
                if ((z3 < contourValue && z4 >= contourValue) || (z3 >= contourValue && z4 < contourValue)) {
                    val t = (contourValue - z3) / (z4 - z3)
                    val xc = i + t
                    val yc = j + 1
                    graphics.drawLine(
                        (xc * width / (width - 1)).toInt(), ((height - yc) * height / (height - 1)),
                        ((xc + xStep) * width / (width - 1)).toInt(), ((height - yc) * height / (height - 1))
                    )
                }
                if ((z2 < contourValue && z4 >= contourValue) || (z2 >= contourValue && z4 < contourValue)) {
                    val t = (contourValue - z2) / (z4 - z2)
                    val xc = i + 1
                    val yc = j + t
                    graphics.drawLine(
                        (xc * width / (width - 1)), ((height - yc) * height / (height - 1)).toInt(),
                        (xc * width / (width - 1)), ((height - (yc + yStep)) * height / (height - 1)).toInt()
                    )
                }
            }
        }
    }

    graphics.color = Color.RED
    graphics.font = Font("Arial", Font.PLAIN, 12)
    trajectories.forEach { (_, trajectory) ->
        val startX = ((startPoint[0] - xlim.first) / (xlim.second - xlim.first) * width).toInt()
        val startY = ((ylim.second - startPoint[1]) / (ylim.second - ylim.first) * height).toInt()
        graphics.drawString("Start (${startPoint[0]}, ${startPoint[1]})", startX, startY)
        trajectory.zipWithNext { a, b ->
            val x1 = ((a[0] - xlim.first) / (xlim.second - xlim.first) * width).toInt()
            val y1 = ((ylim.second - a[1]) / (ylim.second - ylim.first) * height).toInt()
            val x2 = ((b[0] - xlim.first) / (xlim.second - xlim.first) * width).toInt()
            val y2 = ((ylim.second - b[1]) / (ylim.second - ylim.first) * height).toInt()
            graphics.drawLine(x1, y1, x2, y2)
        }
        val endX = ((trajectory.last()[0] - xlim.first) / (xlim.second - xlim.first) * width).toInt()
        val endY = ((ylim.second - trajectory.last()[1]) / (ylim.second - ylim.first) * height).toInt()
        graphics.drawString("End (${trajectory.last()[0]}, ${trajectory.last()[1]})", endX, endY)
    }

    val outputFile = File(imgName ?: "$titlePrefix.png")
    ImageIO.write(image, "png", outputFile)
}

suspend fun plotAsync(
    function: (DoubleArray) -> Double,
    startPoint: DoubleArray,
    xlim: Pair<Double, Double>,
    ylim: Pair<Double, Double>,
    titlePrefix: String,
    trajectories: Map<String, List<DoubleArray>>,
    imgName: String? = null
) = withContext(Dispatchers.Default) {
    plotContourAndTrajectory(function, startPoint, xlim, ylim, titlePrefix, trajectories, imgName)
}

fun asyncAll() = runBlocking {
    val startPoints = mapOf(
        "quadratic" to doubleArrayOf(0.5, -2.0),
        "rosenbrock" to doubleArrayOf(-1.0, 2.0),
        "heavyQuadratic" to doubleArrayOf(0.5, -2.0)
    )

    val functions = listOf(
        Triple("Quadratic Function", ::quadratic, ::quadraticGrad),
        Triple("Rosenbrock Function", ::rosenbrock, ::rosenbrockGrad),
        Triple("Heavy Quadratic Function", ::heavyQuadratic, ::heavyQuadraticGrad)
    )

    val methods = listOf(
        "const descent" to { f: (DoubleArray) -> Double, g: (DoubleArray) -> DoubleArray, p: DoubleArray ->
            gradientDescentWithTrajectory(g, p)
        },
        "dich descent" to { f: (DoubleArray) -> Double, g: (DoubleArray) -> DoubleArray, p: DoubleArray ->
            gradientDescentDichotomyWithTrajectory(f, g, p)
        },
        "golden section descent" to { f: (DoubleArray) -> Double, g: (DoubleArray) -> DoubleArray, p: DoubleArray ->
            gradientDescentGoldenSectionWithTrajectory(f, g, p)
        }
    )

    val jobs = mutableListOf<Deferred<Unit>>()
    for ((name, function, gradFunction) in functions) {
        val startPoint = startPoints[name.split(" ")[0].lowercase(Locale.getDefault())] ?: doubleArrayOf(0.5, -2.0)
        var xLim: Pair<Double, Double>
        var yLim: Pair<Double, Double>
        if (name == "Rosenbrock Function") {
            xLim = Pair(-2.0, 2.0)
            yLim = Pair(-1.0, 3.0)
        } else {
            xLim = Pair(0.0, 3.0)
            yLim =  Pair(-5.0, -1.0)
        }
        for ((methodName, method) in methods) {
            jobs += async {
                val (_, trajectory) = method(function, gradFunction, startPoint)
                plotAsync(
                    function,
                    startPoint,
                    xLim,
                    yLim,
                    "$name ($methodName)",
                    mapOf(methodName.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.getDefault())
                                                        else it.toString() } to trajectory)
                )
            }
        }
    }
    jobs.awaitAll()
}

fun main() {
    asyncAll()
}
