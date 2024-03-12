let model;

async function loadModel() {
  model = await mobilenet.load();
  console.log("Model loaded");
}

async function compareImages() {
  const image1 = document.getElementById("image1");
  const image2 = document.getElementById("image2");
  const imageInput1 = document.getElementById("imageInput1");
  const imageInput2 = document.getElementById("imageInput2");
  const similarityScoreElement = document.getElementById("similarityScore");

  if (!imageInput1.files[0] || !imageInput2.files[0]) {
    alert("Please select two images.");
    return;
  }

  const image1Url = URL.createObjectURL(imageInput1.files[0]);
  const image2Url = URL.createObjectURL(imageInput2.files[0]);

  image1.src = image1Url;
  image2.src = image2Url;

  const img1 = await loadImage(image1Url);
  const img2 = await loadImage(image2Url);

  const features1 = await extractFeatures(img1);
  const features2 = await extractFeatures(img2);

  const similarityScore = calculateSimilarity(features1, features2);
  similarityScoreElement.textContent = similarityScore.toFixed(4);
}

async function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () =>
      resolve(
        tf.browser
          .fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .expandDims(),
      );
    img.onerror = (err) => reject(err);
    img.src = url;
  });
}

async function extractFeatures(image) {
  const features = model.infer(image, "conv_preds");
  return features;
}

function calculateSimilarity(featureVector1, featureVector2) {
  const cosineSimilarity = tf.metrics.cosineSimilarity(
    featureVector1,
    featureVector2,
  );
  return cosineSimilarity.dataSync()[0];
}

window.onload = function () {
  loadModel();
};
