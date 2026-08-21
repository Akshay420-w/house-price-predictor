// Force ONNX Runtime to look in the local folder for WASM assets
if (window.ort && window.ort.env) {
    ort.env.wasm.wasmPaths = './';
    
    // Disable multi-threading and proxy workers to prevent file:/// security blocks
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
}

let session = null;

// Normalization Parameters (Standard Scaler)
const scaler = {
    mean_X: [1525.5559602018186, 5.426, 998.6440240312184, 2.010666666666667, 1.9933333333333334],
    std_X: [492.14552327730075, 2.8552625098228726, 391.56439566892084, 1.4118142779967697, 0.8099108318547018],
    mean_y: 317348.17386082216,
    std_y: 65130.26246675981
};

// Initialize file selector listener
document.addEventListener('DOMContentLoaded', () => {
    const modelSelector = document.getElementById('model-selector');

    if (modelSelector) {
        modelSelector.addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (!file) return;

            try {
                const modelBuffer = await file.arrayBuffer();

                // Create session using WASM execution provider
                session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
                    executionProviders: ['wasm']
                });

                alert("ONNX Model loaded successfully!");
            } catch (error) {
                console.error("Failed to load ONNX model:", error);
                alert("Failed to load local ONNX model file. See console for details.");
            }
        });
    }
});

// Run prediction function
async function predictPrice() {
    if (!session) {
        alert("Please select and load your ONNX model file first.");
        return;
    }

    try {
        const grLivArea = parseFloat(document.getElementById('grLivArea').value) || 0;
        const overallQual = parseFloat(document.getElementById('overallQual').value) || 0;
        const totalBsmtSF = parseFloat(document.getElementById('totalBsmtSF').value) || 0;
        const garageCars = parseFloat(document.getElementById('garageCars').value) || 0;
        const fullBath = parseFloat(document.getElementById('fullBath').value) || 0;

        const rawInputs = [grLivArea, overallQual, totalBsmtSF, garageCars, fullBath];

        // Standardization: (X - mean) / std
        const standardizedInputs = rawInputs.map((val, idx) => (val - scaler.mean_X[idx]) / scaler.std_X[idx]);

        // Input Tensor [1, 5]
        const inputTensor = new ort.Tensor('float32', new Float32Array(standardizedInputs), [1, 5]);

        const inputName = session.inputNames[0];
        const feeds = {};
        feeds[inputName] = inputTensor;

        // Run inference
        const results = await session.run(feeds);

        const outputName = session.outputNames[0];
        const outputTensor = results[outputName];
        const scaledPrediction = outputTensor.data[0];

        // Inverse standardization: (prediction * std) + mean
        const finalPrice = (scaledPrediction * scaler.std_y) + scaler.mean_y;

        const formattedPrice = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(finalPrice);

        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.innerText = `Prediction: ${formattedPrice}`;
        }
    } catch (error) {
        console.error("Inference error:", error);
        alert("An error occurred during prediction calculation.");
    }
}