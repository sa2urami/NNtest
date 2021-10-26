import { WebSocketServer } from 'ws'
import lodash from 'lodash'

const wss = new WebSocketServer({ port: 8080 })

let connected = false

wss.on('connection', ws => {
    console.log('connected')
    if (connected) throw new Error('already connected')
    connected = true
    ws.on('message', values => {
        const arr = JSON.parse(String(values)) as number[][]
        const pixelsLightness = arr.flat(1)
        console.log(pixelsLightness.slice(28, 40))
    })
    ws.on('close', () => (connected = false))
    ws.on('error', err => {
        connected = false
        throw err
    })
})
