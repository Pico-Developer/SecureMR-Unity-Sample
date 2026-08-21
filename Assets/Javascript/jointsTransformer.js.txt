var positionArray;
var matrixArray;
var moleVisibleArray;
var moleCollisions;
var realtimesincestartup;
var moleOnTimings;
var moleOffTimings;
var moleDeadlines;
var score;

let indexArray = [26, 25, 28, 27, 12, 11, 14, 13, 16, 15, 0, 24, 23];
let whackerIndices = [8, 9];
let MAX_MOLES = 5;
let hitThreshold = 0.15;

/**
 * Multiplies two 4x4 matrices (A and B) and stores the result in a new Float32Array.
 *
 * @param {Float32Array} matA The first 4x4 matrix (16 elements).
 * @param {Float32Array} matB The second 4x4 matrix (16 elements).
 * @returns {Float32Array} The resulting 4x4 matrix.
 */
function multiply4x4Matrices(matA, matB) {
    // Both matrices must be 4x4, represented as 16-element Float32Arrays.
    if (matA.length !== 16 || matB.length !== 16) {
        throw new Error("Both matrices must be 4x4 (16 elements).");
    }

    const result = new Float32Array(16);

    for (let i = 0; i < 4; i++) { // Row index for the result matrix
        for (let j = 0; j < 4; j++) { // Column index for the result matrix
            let sum = 0;
            for (let k = 0; k < 4; k++) {
                // The formula for a row-major matrix element at (row, column) is index = row * 4 + column.
                // The standard matrix multiplication formula for an element C[i, j] is
                // sum(A[i, k] * B[k, j]) for all k from 0 to 3.
                sum += matA[i * 4 + k] * matB[k * 4 + j];
            }
            // Store the result element C[i, j] at the row-major index i * 4 + j
            result[i * 4 + j] = sum;
        }
    }

    return result;
}


function createTransformationMatrix(position, scale) {

// 1. Scaling Matrix
    let scaleMatrix = new Float32Array([
        scale[0], 0, 0, 0,
        0, scale[1], 0, 0,
        0, 0, scale[2], 0,
        0, 0, 0, 1
    ]);

    // 2. Translation Matrix
    let translationMatrix = new Float32Array([
        1, 0, 0, position[0], // Translation X in the last column of the first row
        0, 1, 0, position[1], // Translation Y in the last column of the second row
        0, 0, 1, position[2], // Translation Z in the last column of the third row
        0, 0, 0, 1            // The bottom row consists of [0, 0, 0, 1]
    ]);
    // Combine (Scale -> Rotate -> Translate)
    // Order of multiplication for row-major format: M = S * R * T
    let matrix = multiply4x4Matrices(scaleMatrix, translationMatrix);

    return matrix;
}

function copyMatrixToArray(matrix, array, arrayIndex)
{

  for (var j = 0; j < matrix.length; j++)
  {
    array[arrayIndex+j] = matrix[j];
  }

}


let scale = new Float32Array([1.0, -1.0, -1.0]);


for (var i = 0; i < indexArray.length; i++) {
    var index = indexArray[i];
    arrayIndex = index*3;
    let position = new Float32Array([positionArray[arrayIndex], positionArray[arrayIndex+1], positionArray[arrayIndex+2]]);
    let matrix  = createTransformationMatrix(position, scale);
    copyMatrixToArray(matrix, matrixArray, i*16);
}

// moleVisibleArray and moleDeadlines are provided as persistent globals from C++

for (let m = 0; m < MAX_MOLES; m++) {
    if (moleVisibleArray[m] === 0) {
        if (realtimesincestartup[0] >= moleDeadlines[m]) {
            moleVisibleArray[m] = 1;
            moleDeadlines[m] = realtimesincestartup[0] + moleOffTimings[m];
        }
    } else {
        if (realtimesincestartup[0] >= moleDeadlines[m]) {
            moleVisibleArray[m] = 0;
            moleDeadlines[m] = realtimesincestartup[0] + moleOnTimings[m];
        }
    }
}

let whackerPositions = new Float32Array(whackerIndices.length * 3);
for (let w = 0; w < whackerIndices.length; w++) {
    let wi = whackerIndices[w] * 3;
    whackerPositions[w * 3] = positionArray[wi];
    whackerPositions[w * 3 + 1] = positionArray[wi + 1];
    whackerPositions[w * 3 + 2] = positionArray[wi + 2];
}

for (let m = 0; m < MAX_MOLES; m++) {
    if (moleVisibleArray[m] !== 1) continue;
    let base = m * 16;
    let mx = moleCollisions[base + 3];
    let my = moleCollisions[base + 7];
    let mz = moleCollisions[base + 11];
    let hit = false;
    for (let w = 0; w < whackerIndices.length && !hit; w++) {
        let wbase = whackerIndices[w] * 16;
        let wx = matrixArray[wbase + 3];
        let wy = matrixArray[wbase + 7];
        let wz = matrixArray[wbase + 11];
        let dx = wx - mx;
        let dy = wy - my;
        let dz = 0;
        let d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < hitThreshold * hitThreshold) {
            hit = true;
        }
    }
    if (hit) {
        score[0] = score[0] + 1;
        moleVisibleArray[m] = 0;
        moleDeadlines[m] = realtimesincestartup[0] + moleOnTimings[m];
    }
}
