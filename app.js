// Global configuration variables
let session = null;
let scaler = null;

// Initialize and load ONNX Model and Scaler JSON on startup
async function init() {
    try {
        // Fetch feature standardization parameters
        const scalerResponse = await fetch('scaler.json');
        scaler = await scalerResponse.json();

        // Create ONNX Execution Session
        session = await ort.InferenceSession.create('./model.onnx');
        console.log("ONNX Model and Scaler loaded successfully.");
    } catch (error) {
        console.error("Failed to load model resources:", error);
    }
}

// Main execution function called by UI button
async function predictPrice() {
    if (!session || !scaler) {
        alert("Model files are still loading. Please wait a second and try again.");
        return;
    }

    // 1. Extract raw numerical values from input fields
    const grLivArea = parseFloat(document.getElementById('grLivArea').value);
    const overallQual = parseFloat(document.getElementById('overallQual').value);
    const totalBsmtSF = parseFloat(document.getElementById('totalBsmtSF').value);
    const garageCars = parseFloat(document.getElementById('garageCars').value);
    const fullBath = parseFloat(document.getElementById('fullBath').value);

    const rawInputs = [grLivArea, overallQual, totalBsmtSF, garageCars, fullBath];

    // 2. Normalize inputs: (X - Mean) / StdDev
    const normalizedInputs = rawInputs.map((value, index) => {
        return (value - scaler.mean_X[index]) / scaler.std_X[index];
    });

    // 3. Construct Float32Array and ONNX Tensor Object (Batch size: 1, Features: 5)
    const inputData = new Float32Array(normalizedInputs);
    const tensorInput = new ort.Tensor('float32', inputData, [1, 5]);

    // 4. Feed tensor to ONNX Inference Engine
    const feeds = { input: tensorInput };
    const results = await session.run(feeds);

    // 5. Extract prediction output tensor
    const outputTensor = results.output;
    const scaledPrediction = outputTensor.data[0];

    // 6. Inverse Transform prediction back to real USD currency: (Y_scaled * Std_Y) + Mean_Y
    const finalPrice = (scaledPrediction * scaler.std_y) + scaler.mean_y;

    // 7. Update DOM UI Element formatted as USD currency
    document.getElementById('result').innerText = `Prediction: ${formatCurrency(finalPrice)}`;
}

// Utility function to format raw numbers to USD standard currency
function formatCurrency(amount) {
    // Ensure value does not render negative numbers in fringe scenarios
    const safeAmount = Math.max(0, amount);
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0
    }).format(safeAmount);
}

// Run setup initialization when the browser DOM content is fully loaded
window.addEventListener('DOMContentLoaded', init);