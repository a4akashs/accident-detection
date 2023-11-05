// script.js

const imageInput = document.getElementById('imageInput');
const uploadedImage = document.getElementById('uploadedImage');
const webcam = document.getElementById('webcam');
const startWebcamButton = document.getElementById('startWebcam');
const stopWebcamButton = document.getElementById('stopWebcam');
const predictionResult = document.getElementById('predictionResult');
let stream;
let net;

async function loadModel() {
    net = await mobilenet.load();
}

imageInput.addEventListener('change', async function() {
    const file = imageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = async function(e) {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = 'block';
            const img = document.getElementById('uploadedImage');
            const result = await net.classify(img);
            const topPrediction = result[0].className;

            // Simplify the prediction to "accident" or "no accident"
            const isAccident = topPrediction.toLowerCase().includes("accident");
            const predictionText = isAccident ? "Accident" : "No Accident";
            predictionResult.textContent = `Prediction: ${predictionText}`;
        };
        reader.readAsDataURL(file);
    }
});

startWebcamButton.addEventListener('click', async function() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcam.srcObject = stream;
        webcam.style.display = 'block';
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
});

stopWebcamButton.addEventListener('click', function() {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        webcam.srcObject = null;
        webcam.style.display = 'none';
    }
});

loadModel();
