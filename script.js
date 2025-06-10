console.log('Hello TensorFlow');

async function run() {
    console.log("Starte Visualisierung für R1 und R2 ...");

    // 1. Generiere saubere und verrauschte Daten
    const generatedData = generateData(100, 0.05); // Generiere 100 Punkte mit 0.05 Rauschvarianz.
    console.log("Generierte Daten", generatedData);

    // Visualisiere R1 (bestehende Funktion für Clean/Noisy Data)
    visualizeR1(generatedData);

   console.log("R1 - Clean Training Data:", generatedData.clean.train.slice(0, 5));
   console.log("R1 - Clean Test Data:", generatedData.clean.test.slice(0, 5));

    // 2. Trainiere Modell auf sauberen Trainingsdaten
    console.log("Trainiere Modell auf sauberen Daten...");
    const model = createModel(); // Erstelle ein Linearmodell
    const tensorData = convertToTensor(generatedData.clean.train); // Konvertiere die sauberen Trainingsdaten in Tensoren
    const { inputs, labels } = tensorData;

    try {
        await trainModel(model, inputs, labels); // Trainiere das Modell
        console.log("Modell erfolgreich trainiert!");
    } catch (err) {
        console.error("Fehler beim Modelltraining:", err);
        return;
    }

    // 3. Visualisiere Vorhersagen für Trainings- und Testdaten
    console.log("Visualisiere Ergebnisse für R2...");
    visualizeR2(model, generatedData.clean.train, generatedData.clean.test, tensorData);

    // 1. Trainingshafte Daten mit Rauschen abrufen
    const generatedNoisyData = generateData(100, 0.1); // Generiere Daten mit stärkerem Rauschen

    // 2. Trainiere das beste Modell auf verrauschten Daten
    const bestModel = createModel();
    const noisyTensorData = convertToTensor(generatedNoisyData.noisy.train);
    const { inputs: noisyInputs, labels: noisyLabels } = noisyTensorData;

    await trainModel(bestModel, noisyInputs, noisyLabels);

    // 3. Visualisiere Vorhersagen des besten Modells auf verrauschten Daten
    visualizeR3(bestModel, generatedNoisyData.noisy.train, generatedNoisyData.noisy.test, noisyTensorData);

        // 1. Overfitting-Modell auf verrauschten Daten trainieren
        const overfitModel = createModel();
        const overfitNoisyTensorData = convertToTensor(generatedNoisyData.noisy.train);
        const { inputs: overfitNoisyInputs, labels: overfitNoisyLabels } = overfitNoisyTensorData;

        // Verwende die Funktion, um das Overfit-Modell zu trainieren
        await trainOverfitModel(overfitModel, overfitNoisyInputs, overfitNoisyLabels);

        // 2. Visualisiere die Vorhersagen des Overfit-Modells
        visualizeR4(overfitModel, generatedNoisyData.noisy.train, generatedNoisyData.noisy.test, overfitNoisyTensorData);


    console.log("Visualisierung für R1 und R2 abgeschlossen.");
}

document.addEventListener('DOMContentLoaded', run);

// function createModel() {
//     // Create a sequential model
//     const model = tf.sequential();
//
//     // Add a single input layer
//     model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
//
//     // Add an output layer
//     model.add(tf.layers.dense({units: 1, useBias: true}));
//
//     return model;
// }

/**
 * Convert the input data to tensors that we can use for machine
 * learning. The inputs are normalized, but the labels remain unnormalized.
 */
function convertToTensor(data) {
    return tf.tidy(() => {
        // 1. Daten mischen
        tf.util.shuffle(data);

        // 2. Extrahiere Inputs (x) und Labels (y)
        const inputs = data.map(d => d.x);
        const labels = data.map(d => d.y);

        // 3. Konvertiere Inputs zu Tensoren und normalisiere sie
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        // 4. Eingaben normalisieren
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));

        console.log("Original Inputs:", inputs.slice(0, 5)); // Logge erste 5 Eingabewerte
        console.log("Original Labels:", labels.slice(0, 5)); // Logge erste 5 Outputs
        console.log("Normalized Inputs (Tensor):", normalizedInputs.arraySync().slice(0, 5)); // Logge normalisierte Inputs

        return {
            inputs: normalizedInputs, // Normalisierte Inputs
            labels: labelTensor, // Keine Normalisierung der Outputs
            inputMax,
            inputMin,
        };
    });
}

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(0.01), // Angepasste Lernrate
        loss: tf.losses.meanSquaredError, // MSE als Loss
        metrics: ['mse'], // Überwachung von MSE
    });

    const batchSize = 32;
    const epochs = 50;

    console.log("Starte Training...");
    let finalLoss;

    const history = await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Epoch: ${epoch + 1}, Loss: ${logs.loss}, MSE: ${logs.mse}`);
                finalLoss = logs.loss;
            },
        },
    });

    console.log("Final Loss (nach Training):", finalLoss);
    return history;
}
// R4
async function trainOverfitModel(model, inputs, labels) {
    // Kompiliere das Modell mit hoher Lernrate
    model.compile({
        optimizer: tf.train.adam(0.01), // Lernrate für schnelles Training
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 300; // Lange Trainingsdauer, um Overfitting zu provozieren

    console.log("Starte Training des Overfit-Modells...");
    const history = await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}, MSE = ${logs.mse}`);
            },
        },
    });

    console.log("Training des Overfit-Modells abgeschlossen.");
    return history;
}
function testModel(model, inputData, normalizationData) {
    const { inputMax, inputMin } = normalizationData;

    // Generate predictions for test data
    const xsNorm = tf.linspace(0, 1, 100); // Normalized test inputs
    const predictions = tf.tidy(() => {
        // Modellvorhersagen basierend auf normalisierten Eingaben
        return model.predict(xsNorm.reshape([100, 1]));
    });

    // Un-normalize the inputs for visualization
    const unNormXs = xsNorm.mul(inputMax.sub(inputMin)).add(inputMin);

    // Logs außerhalb von tf.tidy()
    console.log("Testdaten (Original):", inputData.slice(0, 5)); // Original-Test-Daten
    console.log("Eingabe-Werte (Un-normalized x):", unNormXs.dataSync().slice(0, 5)); // Un-normalisierte Eingaben

    // Vorhersagen als Array extrahieren und loggen
    predictions.array().then((predArray) => {
        // "Flatten" für bessere Lesbarkeit
        const flatPredictions = predArray.flat();

        // Logs aktualisieren
        console.log("Vorhersagen (Predictions):", flatPredictions.slice(0, 5)); // Zeige die ersten 5 Vorhersagen flach

        // Plotte die Vorhersagen zusammen mit den Testdaten
        const predictedPoints = Array.from(unNormXs.dataSync()).map((x, i) => {
            return { x: x, y: flatPredictions[i] }; // nutze flache Werte
        });

        const originalPoints = inputData.map((d) => ({
            x: d.x,
            y: d.y,
        }));

        // Visualisierung der Ergebnisse
        tfvis.render.scatterplot(
            { name: 'Model Predictions vs Original Data' },
            { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
            {
                xLabel: 'x',
                yLabel: 'y',
                height: 300,
            }
        );
    }).catch((error) => {
        console.error("Fehler bei der Verarbeitung von predictions.array():", error);
    });

    // Manuelles Dispose, um Ressourcen freizugeben
    xsNorm.dispose();
    predictions.dispose();
}
function visualizeR1(generatedData) {
    console.log("Starte Visualisierung für R1...");

    // 1. Extrahiere saubere Trainings- und Testdaten
    const cleanTrainValues = generatedData.clean.train.map(d => ({ x: d.x, y: d.y }));
    const cleanTestValues = generatedData.clean.test.map(d => ({ x: d.x, y: d.y }));

    // 2. Schaue nach vorliegenden verrauschten Trainings- und Testdaten
    const noisyTrainValues = generatedData.noisy.train.map(d => ({ x: d.x, y: d.y }));
    const noisyTestValues = generatedData.noisy.test.map(d => ({ x: d.x, y: d.y }));

    // 3. Visualisiere Clean Data (R1: Clean Data (Train/Test))
    const cleanDataWrapper = document.getElementById("clean-data");
    if (cleanDataWrapper) {
        tfvis.render.scatterplot(
            cleanDataWrapper, // Element `#clean-data`
            { values: [cleanTrainValues, cleanTestValues], series: ["Train", "Test"] },
            {
                xLabel: "x",
                yLabel: "y",
                height: 300,
            }
        );
    } else {
        console.error("Element mit ID `clean-data` nicht gefunden.");
    }

    // 4. Visualisiere Noisy Data (R1: Noisy Data (Train/Test))
    const noisyDataWrapper = document.getElementById("noisy-data");
    if (noisyDataWrapper) {
        tfvis.render.scatterplot(
            noisyDataWrapper, // Element `#noisy-data`
            { values: [noisyTrainValues, noisyTestValues], series: ["Train", "Test"] },
            {
                xLabel: "x",
                yLabel: "y",
                height: 300,
            }
        );
    } else {
        console.error("Element mit ID `noisy-data` nicht gefunden.");
    }

    console.log("Visualisierung für R1 abgeschlossen.");
}
function visualizeR2(model, trainData, testData, normalizationData) {
    console.log("Starte Visualisierung für R2...");

    // **1. Normalisierungsparameter extrahieren**
    const { inputMax, inputMin } = normalizationData;

    try {
        const trainYs = trainData.map(d => d.y); // Labelwerte für Training extrahieren
        const labelMin = Math.min(...trainYs);
        const labelMax = Math.max(...trainYs);

        console.log("labelMin aus Trainingsdaten:", labelMin);
        console.log("labelMax aus Trainingsdaten:", labelMax);

        // **2. Normalisiere Eingabedaten**
        const trainXs = trainData.map(d => d.x);
        const testXs = testData.map(d => d.x);

        const normalizedTrainInputsTensor = tf.tensor2d(trainXs, [trainXs.length, 1])
            .sub(inputMin)
            .div(inputMax.sub(inputMin));
        const normalizedTestInputsTensor = tf.tensor2d(testXs, [testXs.length, 1])
            .sub(inputMin)
            .div(inputMax.sub(inputMin));

        console.log("Original Train Inputs:", trainXs.slice(0, 5));
        console.log("Normalized Train Inputs Tensor:", normalizedTrainInputsTensor.arraySync().slice(0, 5));
        console.log("Normalized Test Inputs Tensor:", normalizedTestInputsTensor.arraySync().slice(0, 5));

        // **3. Berechne Modellvorhersagen**
        const trainPredictions = model.predict(normalizedTrainInputsTensor);
        const testPredictions = model.predict(normalizedTestInputsTensor);

        console.log("Train Predictions Raw Tensor:", trainPredictions.arraySync().slice(0, 5));
        console.log("Test Predictions Raw Tensor:", testPredictions.arraySync().slice(0, 5));

        // **4. Loss (MSE) berechnen**
        const trainLabelsTensor = tf.tensor2d(trainYs, [trainYs.length, 1]);
        const testLabelsTensor = tf.tensor2d(testData.map(d => d.y), [testData.length, 1]);
        const trainMSE = calculateMSE(trainPredictions, trainLabelsTensor);
        const testMSE = calculateMSE(testPredictions, testLabelsTensor);

        console.log("R2 Train MSE:", trainMSE);
        console.log("R2 Test MSE:", testMSE);

        // **5. Denormalisieren der Vorhersagen**
        const denormTrainYs = trainPredictions
            .mul(labelMax - labelMin)
            .add(labelMin)
            .dataSync();
        const denormTestYs = testPredictions
            .mul(labelMax - labelMin)
            .add(labelMin)
            .dataSync();

        console.log("Denormalized Train Predictions:", Array.from(denormTrainYs).slice(0, 5));
        console.log("Denormalized Test Predictions:", Array.from(denormTestYs).slice(0, 5));

        // **6. Scatterplot-Daten**
        const predictedTrainPoints = trainXs.map((x, i) => ({
            x: x,
            y: denormTrainYs[i],
        }));
        const predictedTestPoints = testXs.map((x, i) => ({
            x: x,
            y: denormTestYs[i],
        }));

        const originalTrainPoints = trainData.map(d => ({ x: d.x, y: d.y }));
        const originalTestPoints = testData.map(d => ({ x: d.x, y: d.y }));

        // **7. Visualisiere Trainingsdaten**
        const trainWrapper = document.getElementById("prediction-clean-train");
        const testWrapper = document.getElementById("prediction-clean-test");

        if (trainWrapper) {
            tfvis.render.scatterplot(
                trainWrapper,
                { values: [originalTrainPoints, predictedTrainPoints], series: ["Original", "Predicted"] },
                { xLabel: "x", yLabel: "y", height: 300 }
            );
        } else {
            console.error("Element #prediction-clean-train nicht gefunden.");
        }

        if (testWrapper) {
            tfvis.render.scatterplot(
                testWrapper,
                { values: [originalTestPoints, predictedTestPoints], series: ["Original", "Predicted"] },
                { xLabel: "x", yLabel: "y", height: 300 }
            );
        } else {
            console.error("Element #prediction-clean-test nicht gefunden.");
        }

        // **8. Loss-Werte unter den Diagrammen anzeigen**
        const trainLossElement = document.getElementById('loss-clean-train');
        const testLossElement = document.getElementById('loss-clean-test');

        if (trainLossElement && testLossElement) {
            trainLossElement.innerText = `Loss (MSE): ${trainMSE.toFixed(4)}`;
            testLossElement.innerText = `Loss (MSE): ${testMSE.toFixed(4)}`;
        } else {
            console.error("Loss-Elemente für R2 (#loss-clean-train, #loss-clean-test) nicht gefunden.");
        }
        renderChartsWithChartJS(model, trainData, testData, 'loss-chart-r2', 'mse-chart-r2');

        // **9. Speicher freigeben**
        normalizedTrainInputsTensor.dispose();
        normalizedTestInputsTensor.dispose();
        trainPredictions.dispose();
        testPredictions.dispose();
        trainLabelsTensor.dispose();
        testLabelsTensor.dispose();



        console.log("Visualisierung für R2 erfolgreich abgeschlossen.");

    } catch (err) {
        console.error("Fehler in visualizeR2:", err.message);
    }
}


// Hilfsfunktion zur Berechnung von MSE
function calculateMSE(predictions, labels) {
    return tf.tidy(() => {
        const error = tf.sub(predictions, labels); // Fehler berechnen
        const squaredError = tf.square(error);    // Quadriere den Fehler
        const meanSquaredError = tf.mean(squaredError); // Durchschnitt berechnen
        return meanSquaredError.dataSync()[0];   // MSE-Wert zurückgeben
    });
}

async function visualizeR3(bestModel, noisyTrainData, noisyTestData, normalizationData) {
    console.log("Starte Visualisierung für R3 (Best-Fit-Modell)...");

    // **Extrahiere Normalisierungsparameter**
    const { inputMax, inputMin } = normalizationData;

    try {
        // **1. Berechnung von Min/Max der Labels aus verrauschten Trainingsdaten**
        const trainYsNoisy = noisyTrainData.map(d => d.y);
        const labelMin = Math.min(...trainYsNoisy);
        const labelMax = Math.max(...trainYsNoisy);

        console.log("labelMin aus verrauschten Trainingsdaten:", labelMin);
        console.log("labelMax aus verrauschten Trainingsdaten:", labelMax);

        // **2. Normalisiere Trainings- und Testdaten**
        const trainXsNoisy = noisyTrainData.map(d => d.x);
        const testXsNoisy = noisyTestData.map(d => d.x);

        const normalizedTrainInputsTensor = tf.tensor2d(trainXsNoisy, [trainXsNoisy.length, 1])
            .sub(inputMin)
            .div(inputMax.sub(inputMin));
        const normalizedTestInputsTensor = tf.tensor2d(testXsNoisy, [testXsNoisy.length, 1])
            .sub(inputMin)
            .div(inputMax.sub(inputMin));

        // **3. Modellvorhersagen**
        const trainPredictions = bestModel.predict(normalizedTrainInputsTensor);
        const testPredictions = bestModel.predict(normalizedTestInputsTensor);

        // **4. Loss (MSE) berechnen**
        const trainLabelsTensor = tf.tensor2d(trainYsNoisy, [trainYsNoisy.length, 1]);
        const testLabelsTensor = tf.tensor2d(noisyTestData.map(d => d.y), [noisyTestData.length, 1]);

        const trainMSE = calculateMSE(trainPredictions, trainLabelsTensor);
        const testMSE = calculateMSE(testPredictions, testLabelsTensor);

        console.log("R3 Train MSE:", trainMSE);
        console.log("R3 Test MSE:", testMSE);

        // **5. Vorhersagen denormalisieren**
        const denormTrainYs = trainPredictions
            .mul(labelMax - labelMin)
            .add(labelMin)
            .dataSync();
        const denormTestYs = testPredictions
            .mul(labelMax - labelMin)
            .add(labelMin)
            .dataSync();

        console.log("Denormalized Noisy Train Predictions:", Array.from(denormTrainYs).slice(0, 5));
        console.log("Denormalized Noisy Test Predictions:", Array.from(denormTestYs).slice(0, 5));

        // **6. Erstelle Scatterplot-Daten**
        const predictedTrainPoints = trainXsNoisy.map((x, i) => ({
            x: x,
            y: denormTrainYs[i],
        }));

        const predictedTestPoints = testXsNoisy.map((x, i) => ({
            x: x,
            y: denormTestYs[i],
        }));

        const originalTrainPoints = noisyTrainData.map(d => ({ x: d.x, y: d.y }));
        const originalTestPoints = noisyTestData.map(d => ({ x: d.x, y: d.y }));

        console.log("Predicted Noisy Train Points:", predictedTrainPoints.slice(0, 5));
        console.log("Predicted Noisy Test Points:", predictedTestPoints.slice(0, 5));

        // **7. Visualisiere Daten für Trainingsdaten**
        const trainWrapper = document.getElementById("prediction-noisy-train");
        if (trainWrapper) {
            tfvis.render.scatterplot(
                trainWrapper,
                { values: [originalTrainPoints, predictedTrainPoints], series: ["Original", "Predicted"] },
                { xLabel: "x", yLabel: "y", height: 300 }
            );
        } else {
            console.error("Element für Trainingsdaten (#prediction-noisy-train) nicht gefunden.");
        }

        // **8. Visualisiere Daten für Testdaten**
        const testWrapper = document.getElementById("prediction-noisy-test");
        if (testWrapper) {
            tfvis.render.scatterplot(
                testWrapper,
                { values: [originalTestPoints, predictedTestPoints], series: ["Original", "Predicted"] },
                { xLabel: "x", yLabel: "y", height: 300 }
            );
        } else {
            console.error("Element für Testdaten (#prediction-noisy-test) nicht gefunden.");
        }

        // **9. Zeige MSE als Text unter den Diagrammen an**
        const trainLossElement = document.getElementById('loss-noisy-train');
        const testLossElement = document.getElementById('loss-noisy-test');

        if (trainLossElement && testLossElement) {
            trainLossElement.innerText = `Loss (MSE): ${trainMSE.toFixed(4)}`; // Train-Loss anzeigen
            testLossElement.innerText = `Loss (MSE): ${testMSE.toFixed(4)}`;   // Test-Loss anzeigen
        } else {
            console.error("Loss-Elemente (#loss-noisy-train, #loss-noisy-test) nicht gefunden.");
        }
        renderChartsWithChartJS(bestModel, noisyTrainData, noisyTestData, 'loss-chart-r3', 'mse-chart-r3');

        // **10. Ressourcen freigeben**
        trainPredictions.dispose();
        testPredictions.dispose();
        normalizedTrainInputsTensor.dispose();
        normalizedTestInputsTensor.dispose();
        trainLabelsTensor.dispose();
        testLabelsTensor.dispose();

        // Neue Loss- und MSE-Diagramme hinzufügen



        console.log("Visualisierung für R3 abgeschlossen.");
    } catch (err) {
        console.error("Fehler in visualizeR3:", err.message);
    }
}
// R4
function denormalizePredictions(predictions, labelMin, labelMax) {
    return predictions
        .mul(labelMax - labelMin) // Zurückskalieren auf den ursprünglichen Wertebereich
        .add(labelMin);
}
async function visualizeR4(overfitModel, noisyTrainData, noisyTestData, normalizationData) {
    console.log("Starte Visualisierung für R4 (Overfit-Modell)...");

    const { inputMax, inputMin } = normalizationData;

    // Berechne Min/Max-Werte der Labels
    const trainYsNoisy = noisyTrainData.map(d => d.y);
    const labelMin = Math.min(...trainYsNoisy);
    const labelMax = Math.max(...trainYsNoisy);

    // Normalisiere Inputs für Trainings- und Testdaten
    const trainXsNoisy = noisyTrainData.map(d => d.x);
    const testXsNoisy = noisyTestData.map(d => d.x);
    const normalizedTrainInputsTensor = tf.tensor2d(trainXsNoisy, [trainXsNoisy.length, 1])
        .sub(inputMin)
        .div(inputMax.sub(inputMin));
    const normalizedTestInputsTensor = tf.tensor2d(testXsNoisy, [testXsNoisy.length, 1])
        .sub(inputMin)
        .div(inputMax.sub(inputMin));

    // Berechne Vorhersagen des Overfit-Modells
    const trainPredictions = overfitModel.predict(normalizedTrainInputsTensor);
    const testPredictions = overfitModel.predict(normalizedTestInputsTensor);

    // Test-Labels als Tensor
    const trainLabelsTensor = tf.tensor2d(trainYsNoisy, [trainYsNoisy.length, 1]);
    const testLabelsTensor = tf.tensor2d(noisyTestData.map(d => d.y), [noisyTestData.length, 1]);

    // Berechne MSE (Loss) für Trainings- und Testdaten
    const trainMSE = calculateMSE(trainPredictions, trainLabelsTensor);
    const testMSE = calculateMSE(testPredictions, testLabelsTensor);

    console.log("Train MSE:", trainMSE);
    console.log("Test MSE:", testMSE);

    // Daten normalisieren für Darstellung
    const denormTrainYs = trainPredictions
        .mul(labelMax - labelMin)
        .add(labelMin)
        .dataSync();
    const denormTestYs = testPredictions
        .mul(labelMax - labelMin)
        .add(labelMin)
        .dataSync();

    // Daten für Scatterplot erstellen
    const predictedTrainPoints = trainXsNoisy.map((x, i) => ({ x: x, y: denormTrainYs[i] }));
    const predictedTestPoints = testXsNoisy.map((x, i) => ({ x: x, y: denormTestYs[i] }));
    const originalTrainPoints = noisyTrainData.map(d => ({ x: d.x, y: d.y }));
    const originalTestPoints = noisyTestData.map(d => ({ x: d.x, y: d.y }));

    // Visualisierung für Trainingsdaten
    const trainWrapper = document.getElementById("overfit-train");
    tfvis.render.scatterplot(
        trainWrapper,
        { values: [originalTrainPoints, predictedTrainPoints], series: ["Original", "Predicted"] },
        { xLabel: "x", yLabel: "y", height: 300 }
    );

    // Visualisierung für Testdaten
    const testWrapper = document.getElementById("overfit-test");
    tfvis.render.scatterplot(
        testWrapper,
        { values: [originalTestPoints, predictedTestPoints], series: ["Original", "Predicted"] },
        { xLabel: "x", yLabel: "y", height: 300 }
    );

    // Zeige MSE unter den Diagrammen an
    const trainLossElement = document.getElementById('loss-overfit-train');
    const testLossElement = document.getElementById('loss-overfit-test');

    if (trainLossElement && testLossElement) {
        trainLossElement.innerText = `Loss (MSE): ${trainMSE.toFixed(4)}`; // Zeigt Train-MSE an
        testLossElement.innerText = `Loss (MSE): ${testMSE.toFixed(4)}`;   // Zeigt Test-MSE an
    } else {
        console.error("Loss-Elemente (HTML) wurden nicht gefunden.");
    }
    // Neue Loss- und MSE-Diagramme hinzufügen
    renderChartsWithChartJS(overfitModel, noisyTrainData, noisyTestData, 'loss-chart-r4', 'mse-chart-r4');


    // Ressourcen freigeben
    trainPredictions.dispose();
    testPredictions.dispose();
    normalizedTrainInputsTensor.dispose();
    normalizedTestInputsTensor.dispose();
    trainLabelsTensor.dispose();
    testLabelsTensor.dispose();
    console.log("Visualisierung für R4 abgeschlossen.");
}
async function runR4() {
    console.log("Starte überfitting Modell-Training für R4...");

    // Generiere verrauschte Daten (z. B. mit 0.2 Rauschvarianz)
    const generatedNoisyData = generateData(20, 0.5);

    // Konvertiere Trainingsdaten in Tensoren
    const noisyTensorData = convertToTensor(generatedNoisyData.noisy.train);
    const { inputs: noisyInputs, labels: noisyLabels } = noisyTensorData;

    // Erstelle und trainiere Overfit-Modell
    const overfitModel = createOverfitModel();
    await trainOverfitModel(overfitModel, noisyInputs, noisyLabels);

    // Visualisiere Ergebnisse von R4
    visualizeR4(
        overfitModel,
        generatedNoisyData.noisy.train,
        generatedNoisyData.noisy.test,
        noisyTensorData
    );
}
function calculateMSE(predictions, labels) {
    // Berechne den MSE mit TensorFlow.js Operationen
    return tf.tidy(() => {
        const error = tf.sub(predictions, labels); // Fehler: Vorhersagen - Labels
        const squaredError = tf.square(error); // Fehler quadrieren
        const meanSquaredError = tf.mean(squaredError); // Durchschnitt berechnen
        return meanSquaredError.dataSync()[0]; // Skalarwert zurückgeben
    });
}
async function visualizePredictionAndLoss(sectionIdPrefix, model, trainData, testData, normalizationData) {
    console.log(`Visualisierung für ${sectionIdPrefix} gestartet...`);

    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    // Extrahiere Eingabedaten (x) und Labels (y)
    const trainXs = trainData.map(d => d.x);
    const testXs = testData.map(d => d.x);
    const trainYs = trainData.map(d => d.y);
    const testYs = testData.map(d => d.y);

    // Normalisiere Eingabedaten
    const normalizedTrainInputsTensor = tf.tensor2d(trainXs, [trainXs.length, 1])
        .sub(inputMin)
        .div(inputMax.sub(inputMin));
    const normalizedTestInputsTensor = tf.tensor2d(testXs, [testXs.length, 1])
        .sub(inputMin)
        .div(inputMax.sub(inputMin));

    // Berechne Vorhersagen des Modells
    const trainPredictions = model.predict(normalizedTrainInputsTensor);
    const testPredictions = model.predict(normalizedTestInputsTensor);

    // Erstelle Tensoren für die echten Labels
    const trainLabelsTensor = tf.tensor2d(trainYs, [trainYs.length, 1]);
    const testLabelsTensor = tf.tensor2d(testYs, [testYs.length, 1]);

    // Berechne MSE für Training und Test
    const trainMSE = calculateMSE(trainPredictions, trainLabelsTensor);
    const testMSE = calculateMSE(testPredictions, testLabelsTensor);

    console.log(`${sectionIdPrefix} Train MSE: ${trainMSE}`);
    console.log(`${sectionIdPrefix} Test MSE: ${testMSE}`);

    // Denormalisiere Vorhersagen, damit sie im Plot dargestellt werden können
    const denormTrainYs = trainPredictions
        .mul(labelMax - labelMin)
        .add(labelMin)
        .dataSync();
    const denormTestYs = testPredictions
        .mul(labelMax - labelMin)
        .add(labelMin)
        .dataSync();

    // Originaldaten und Vorhersagen für Scatterplots
    const predictedTrainPoints = trainXs.map((x, i) => ({ x: x, y: denormTrainYs[i] }));
    const predictedTestPoints = testXs.map((x, i) => ({ x: x, y: denormTestYs[i] }));
    const originalTrainPoints = trainData.map(d => ({ x: d.x, y: d.y }));
    const originalTestPoints = testData.map(d => ({ x: d.x, y: d.y }));

    // Visualisierung für Trainingsdaten
    const trainWrapper = document.getElementById(`${sectionIdPrefix}-train`);
    tfvis.render.scatterplot(
        trainWrapper,
        { values: [originalTrainPoints, predictedTrainPoints], series: ["Original", "Predicted"] },
        { xLabel: "x", yLabel: "y", height: 300 }
    );

    // Visualisierung für Testdaten
    const testWrapper = document.getElementById(`${sectionIdPrefix}-test`);
    tfvis.render.scatterplot(
        testWrapper,
        { values: [originalTestPoints, predictedTestPoints], series: ["Original", "Predicted"] },
        { xLabel: "x", yLabel: "y", height: 300 }
    );

    // MSE unter den Diagrammen anzeigen
    document.getElementById(`loss-${sectionIdPrefix}-train`).innerText = `Loss (MSE): ${trainMSE.toFixed(4)}`;
    document.getElementById(`loss-${sectionIdPrefix}-test`).innerText = `Loss (MSE): ${testMSE.toFixed(4)}`;

    // Ressourcen freigeben
    trainPredictions.dispose();
    testPredictions.dispose();
    normalizedTrainInputsTensor.dispose();
    normalizedTestInputsTensor.dispose();
    trainLabelsTensor.dispose();
    testLabelsTensor.dispose();
    console.log(`Visualisierung für ${sectionIdPrefix} abgeschlossen.`);
}
function renderChartsWithChartJS(model, trainData, testData, lossChartId, mseChartId) {
    console.log(`Erstelle Diagramme: ${lossChartId} und ${mseChartId}`);

    // 1. Loss- und MSE-Werte initialisieren
    const trainLossValues = [];
    const testLossValues = [];
    const epochs = 50;

    // 2. Modelltraining mit Loss-Werten sammeln
    model.fit(
        tf.tensor2d(trainData.map(d => d.x), [trainData.length, 1]),
        tf.tensor2d(trainData.map(d => d.y), [trainData.length, 1]),
        {
            epochs: epochs,
            shuffle: true,
            validationData: [
                tf.tensor2d(testData.map(d => d.x), [testData.length, 1]),
                tf.tensor2d(testData.map(d => d.y), [testData.length, 1])
            ],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: Loss: ${logs.loss}, Val Loss: ${logs.val_loss}`);
                    trainLossValues.push(logs.loss); // Trainings-Loss hinzufügen
                    testLossValues.push(logs.val_loss); // Test-Loss hinzufügen
                }
            }
        }
    ).then(() => {
        // 3. Erstellen des Loss-Diagramms
        const epochLabels = Array.from({ length: epochs }, (_, i) => i + 1);

        const lossCanvas = document.getElementById(lossChartId);
        if (!lossCanvas) {
            console.error(`Canvas mit ID: ${lossChartId} nicht gefunden!`);
            return;
        }

        new Chart(lossCanvas, {
            type: 'line',
            data: {
                labels: epochLabels, // Epochen als X-Achse
                datasets: [
                    {
                        label: 'Trainings-Loss',
                        data: trainLossValues,
                        borderColor: 'rgba(0, 0, 255, 1)', // Komplett deckendes Blau für Linienrahmen
                        backgroundColor: 'rgba(0, 0, 255, 0.2)', // Abgeschwächtes Blau im Hintergrund
                        fill: true // Füllt den Bereich unter der Linie
                    },
                    {
                        label: 'Test-Loss',
                        data: testLossValues,
                        borderColor: 'rgba(255, 165, 0, 1)', // Komplett deckendes Orange für Linienrahmen
                        backgroundColor: 'rgba(255, 165, 0, 0.2)', // Abgeschwächtes Orange im Hintergrund
                        fill: true // Füllt den Bereich unter der Linie
                    }

                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Epochen' }
                    },
                    y: {
                        title: { display: true, text: 'Loss' },
                        beginAtZero: true
                    }
                }
            }
        });

        // Berechnung der Vorhersagen und MSE-Werte
        const trainXs = trainData.map(d => d.x);
        const trainYs = trainData.map(d => d.y);
        const testXs = testData.map(d => d.x);
        const testYs = testData.map(d => d.y);

        const trainPredictions = model.predict(tf.tensor2d(trainXs, [trainXs.length, 1]));
        const testPredictions = model.predict(tf.tensor2d(testXs, [testXs.length, 1]));

        const trainMSE = calculateMSE(trainPredictions, tf.tensor2d(trainYs, [trainYs.length, 1]));
        const testMSE = calculateMSE(testPredictions, tf.tensor2d(testYs, [testYs.length, 1]));

        console.log(`Train MSE: ${trainMSE}, Test MSE: ${testMSE}`);

        // Bereinigung der Tensoren
        trainPredictions.dispose();
        testPredictions.dispose();

        // 4. MSE-Balkendiagramm erstellen
        const mseCanvas = document.getElementById(mseChartId);
        if (!mseCanvas) {
            console.error(`Canvas mit ID: ${mseChartId} nicht gefunden!`);
            return;
        }

        new Chart(mseCanvas, {
            type: 'bar',
            data: {
                labels: ['Train', 'Test'],
                datasets: [{
                    label: 'MSE',
                    data: [trainMSE, testMSE],
                    backgroundColor: ['rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)'], // Abgeschwächte Farben
                    borderColor: ['rgba(0, 0, 255, 1)', 'rgba(255, 165, 0, 1)'], // Dunklere Farben für Rand
                    borderWidth: 1

                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Datensatz' }
                    },
                    y: {
                        title: { display: true, text: 'MSE' },
                        beginAtZero: true
                    }
                }
            }
        });
    }).catch((err) => {
        console.error("Fehler beim Erstellen der Diagramme:", err);
    });
}