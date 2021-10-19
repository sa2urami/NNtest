//@ts-check
import { WebSocketServer } from 'ws'

const wss = new WebSocketServer({ port: 8080 })

wss.on('connection', () => {
    wss.on('message', (message) => {
        console.log('got message', message)
        let values = new Array(10).fill(null).map(() => Math.random())
        wss.send('message', JSON.stringify(values))
    })
})
