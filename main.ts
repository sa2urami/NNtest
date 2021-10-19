const sigmoid = (count: number) => (1 + Math.E ** count) ** -1
const nodes: Node[][] = []

class Neuron {
    value: number
    weighs: number[]
}

class Layer {
    //activationFunction: string
    nodes: Neuron[]
    bias: number[]
    constructor(public size: number, public nextLayer?: Layer) {}
}
let prevLayer: undefined | Layer
let layersIndex: Layer[] = [768, 60, 60, 10]
    .reverse()
    .map((num) => {
        prevLayer = new Layer(num, prevLayer)
        return prevLayer
    })
    .reverse()

const init = () => {
    for (const index of layersIndex) {
        for (let num = 0; num < index.size; num++) index.bias[num] = -10
        if (index === layersIndex.slice(-1)[0]) continue
        for (let num = 0; num < index.size; num++) {
            for (let num1 = 0; num1 < index.nextLayer.size; num1++) {
                index.nodes[num].weighs[num1] = (Math.random() - 0.5) * 10
            }
        }
    }
}

const activation = (layer: Layer) => {
    if (layer === layersIndex.slice(-1)[0]) return
    for (let num = 0; num < layer.nextLayer.size; num++) {
        let buf: number = 0
        for (let num1 = 0; num1 < layer.size; num1++) {
            buf +=
                layer.nextLayer.nodes[num1].value *
                layer.nextLayer.nodes[num1].weighs[num]
        }
        buf += layer.bias[num]
        buf = sigmoid(buf)
        layer.nextLayer.nodes[num].value = buf
    }
}
init()
for (let num = 0; num < 768; num++)
    layersIndex[0].nodes[num].value = Math.random()
activation(layersIndex[0])
