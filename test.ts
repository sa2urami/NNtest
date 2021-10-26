import { promisify } from 'util'
import lodash from 'lodash'
import getPixelsCb from 'get-pixels'
import { writeFileSync } from 'fs'

const getPixels = promisify(getPixelsCb)

const path = 'train/000014-num1.png'

const { data: pixelsData } = await getPixels(path)
const pixelsLightness = lodash.chunk(pixelsData, 4).map(([f]) => f / 255)

writeFileSync('output.json', JSON.stringify(lodash.chunk(pixelsLightness, 28)), 'utf-8')
