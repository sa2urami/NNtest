/* eslint-disable curly */
/* eslint-disable zardoy-config/@typescript-eslint/no-for-in-array, zardoy-config/@typescript-eslint/ban-ts-comment */
import { promises } from 'fs'
import { promisify } from 'util'
import getPixelsCb from 'get-pixels'
import lodash from 'lodash'

import { writeJsonFile } from 'typed-jsonfile'
import { WebSocketServer } from 'ws'

const sigmoid = (count: number) => 1 / (1 + Math.exp(-1 * count))
const dsigmoid = (x: number) => x * (1 - x)
const nodes: Node[][] = []
const learningRate = 0.001
class Neuron {
    value = 0
    weighs: number[] = []
}

class Layer {
    // activationFunction: string
    bias: number[] = []
    nodes: Neuron[]
    constructor(public size: number, public nextLayer?: Layer) {
        this.nodes = new Array(size).fill(null).map(() => new Neuron())
    }
}

let prevLayer: undefined | Layer
const layersIndex: Layer[] = [768, 512, 128, 32, 10]
    .reverse()
    .map(num => {
        prevLayer = new Layer(num, prevLayer)
        return prevLayer
    })
    .reverse()

const init = () => {
    for (const layer of layersIndex) {
        for (let num = 0; num < layer.size; num++) layer.bias[num] = Math.random() * 2 - 1
        if (layer.nextLayer === undefined) continue
        for (let num = 0; num < layer.size; num++) {
            layer.nodes[num] = new Neuron()
            for (let num1 = 0; num1 < layer.nextLayer.size; num1++) layer.nodes[num]!.weighs[num1] = Math.random() * 2 - 1
        }
    }
}

const activation = (layer: Layer) => {
    if (layer.nextLayer === undefined) return
    for (let num = 0; num < layer.nextLayer.size; num++) {
        let buf = 0
        lodash.times(layer.size, num1 => {
            buf += layer.nodes[num1]!.value * layer.nodes[num1]!.weighs[num]!
        })

        buf += layer.nextLayer.bias[num]
        const input = buf
        buf = sigmoid(buf)
        layer.nextLayer.nodes[num]!.value = buf
    }

    activation(layer.nextLayer)
}
let OverallError = 0
let n = 0
const backpropagation = (target: number[]) => {
    let errors: number[] = []
    let buff = 0
    for (let i = 0; i < layersIndex[layersIndex.length - 1]!.nodes.length; i++) {
        errors[i] = target[i]! - layersIndex.at(-1)!.nodes[i]!.value
        buff += errors[i]! * errors[i]!
    }
    OverallError = (OverallError * n + buff) / (n + 1)
    n++
    if (n > 1000) {
        n = 0
        yes = 0
        no = 0
    }
    console.log(OverallError)
    for (let k = layersIndex.length - 2; k >= 0; k--) {
        const l = layersIndex[k]!
        const l1 = layersIndex[k + 1]!
        const ErrorsNext: number[] = []
        const gradients: number[] = []
        for (let i = 0; i < l1.nodes.length; i++) {
            gradients[i] = errors[i]! * dsigmoid(layersIndex[k + 1].nodes[i].value)
            gradients[i] *= learningRate
        }

        const deltas: number[][] = []
        for (let i = 0; i < l1.nodes.length; i++) {
            deltas[i] = []
            for (let j = 0; j < l.nodes.length; j++) deltas[i][j] = gradients[i] * l.nodes[j].value
        }

        for (let i = 0; i < l.nodes.length; i++) {
            ErrorsNext[i] = 0
            for (let j = 0; j < l1.nodes.length; j++) ErrorsNext[i] += l.nodes[i].weighs[j] * errors[j]
        }

        errors = ErrorsNext
        const weightsNew: number[][] = []
        for (let i = 0; i < l1.nodes.length; i++) {
            for (let j = 0; j < l.nodes.length; j++) {
                if (weightsNew[j] === undefined) weightsNew[j] = []

                weightsNew[j][i] = l.nodes[j].weighs[i] + deltas[i][j]
            }
        }

        for (let i = 0; i < l1.nodes.length; i++)
            for (let j = 0; j < l.nodes.length; j++) {
                l.nodes[j].weighs[i] = weightsNew[j][i]
            }

        for (let i = 0; i < l1.nodes.length; i++) l1.bias[i] += gradients[i]
    }
}

init()

// for (let num = 0; num < 768; num++) layersIndex[0].nodes[num].value = Math.random()
// activation(layersIndex[0])

// writeJsonFile(
//     'output.json',
//     layersIndex.slice(-1)[0].nodes.map(({ value }) => value),
// )

const wss = new WebSocketServer({ port: 8080 })

let connected = false

let nextTask: (() => void) | undefined

const oneData = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00392156862745098, 0.6588235294117647, 0.9490196078431372, 0.10980392156862745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0392156862745098, 0.8941176470588236, 0.996078431372549, 0.39215686274509803, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7450980392156863, 0.996078431372549, 0.47843137254901963, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3254901960784314, 0.996078431372549, 0.6352941176470588, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.9725490196078431, 0.09803921568627451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 1, 0.996078431372549, 0.403921568627451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.42745098039215684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.42745098039215684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.42745098039215684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 1, 0.996078431372549, 0.42745098039215684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.42745098039215684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.24705882352941178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.10980392156862745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.10980392156862745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.13725490196078433, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.11372549019607843, 0.996078431372549, 0.996078431372549, 0.42745098039215684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.023529411764705882, 0.8313725490196079, 0.996078431372549, 0.42745098039215684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.796078431372549, 0.996078431372549, 0.6980392156862745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6078431372549019, 0.996078431372549, 0.7450980392156863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.12549019607843137, 0.7803921568627451, 0.40784313725490196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

wss.on('connection', ws => {
    if (connected) throw new Error('already connected')
    connected = true
    ws.on('message', values => {
        nextTask = () => {
            let pixelsLightness: number[]
            if (String(values) === 'true') {
                pixelsLightness = oneData.flat(1)
            } else {
                const arr = JSON.parse(String(values)) as number[][]
                pixelsLightness = arr.flat(1)
            }
            // const pixelsLightness = testData.flat(1)

            for (const index in layersIndex[0].nodes) layersIndex[0].nodes[index].value = pixelsLightness[index]
            activation(layersIndex[0])

            const sendData = layersIndex.at(-1)!.nodes.map(({ value }) => value)
            ws.send(JSON.stringify(sendData))
        }
    })
    ws.on('close', () => (connected = false))
    ws.on('error', err => {
        connected = false
        throw err
    })
})
let yes = 0
let no = 0
// @ts-expect-error
const pictures = await promises.readdir('./train')
for (const pictureName of pictures) {
    const number = +/(\d).png/.exec(pictureName)![1]

    const getPixels = promisify(getPixelsCb)

    // @ts-expect-error
    const { data: pixelsData } = await getPixels('./train/050000-num3.png')

    const pixelsLightness = lodash.chunk(pixelsData, 4).map(([f]) => f / 255)
    for (const index in layersIndex[0].nodes) layersIndex[0].nodes[index].value = pixelsLightness[index]

    activation(layersIndex[0])
    const buf = lodash.times(layersIndex.at(-1)!.nodes.length, () => 0)
    buf[number] = 1
    let number1 = 0
    for (let i = 0; i < 10; i++) {
        let buf = 0
        if (layersIndex[layersIndex.length - 1].nodes[i].value > buf) {
            buf = layersIndex[layersIndex.length - 1].nodes[i].value
            number1 = i
        }
        if (number1 == number) yes++
        else no++
        console.log(yes / n)
        console.log(no / n)
    }
    backpropagation(buf)
    if (nextTask) {
        nextTask()
        nextTask = undefined
    }
}
