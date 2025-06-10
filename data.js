// Funktion, die die Ground-Truth-Funktion berechnet
function groundTruthFunction(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

// Hilfsfunktion: Normalverteiltes Rauschen mit Mittelwert 0 und Standardabweichung sqrt(varianz)
function gaussianNoise(mean = 0, variance = 0.05) {
    const stdDev = Math.sqrt(variance);
    let u = 0, v = 0;
    while (u === 0) u = Math.random(); // Verhindert log(0)
    while (v === 0) v = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdDev + mean;
}

// Funktion zur Datengenerierung
function generateData(N = 100, noiseVariance = 0.05) {
    console.log(`Starte Datengenerierung mit N=${N} und noiseVariance=${noiseVariance}`);

    // 1. Zufällige x-Werte in [-2, 2]
    const xs = Array.from({ length: N }, () => Math.random() * 4 - 2);
    console.log(`Generierte x-Werte:`, xs.slice(0, 10), '...'); // Nur die ersten 10 fürs Debugging

    // 2. Berechne y = f(x) ohne Rauschen
    const ysClean = xs.map(x => groundTruthFunction(x));
    console.log(`Berechnete saubere y-Werte (Ground Truth):`, ysClean.slice(0, 10), '...');

    // 3. Berechne y = f(x) mit Rauschen (nur auf y!)
    const ysNoisy = ysClean.map(y => y + gaussianNoise(0, noiseVariance));
    console.log(`Berechnete verrauschte y-Werte:`, ysNoisy.slice(0, 10), '...');

    // 4. Kombiniere x und y zu Objekten
    const dataClean = xs.map((x, i) => ({ x, y: ysClean[i] }));
    const dataNoisy = xs.map((x, i) => ({ x, y: ysNoisy[i] }));
    console.log(`Beispiel eines unverrauschten Datenpunkts:`, dataClean[0]);
    console.log(`Beispiel eines verrauschten Datenpunkts:`, dataNoisy[0]);

    // 5. Shuffle
    console.log('Mische die Daten...');
    tf.util.shuffle(dataClean);
    tf.util.shuffle(dataNoisy);
    console.log(`Erster Datensatz nach Shuffle (clean):`, dataClean[0]);
    console.log(`Erster Datensatz nach Shuffle (noisy):`, dataNoisy[0]);

    // 6. Split in Trainings- und Testdaten
    const trainSize = Math.floor(N / 2);
    const result = {
        clean: {
            train: dataClean.slice(0, trainSize),
            test: dataClean.slice(trainSize)
        },
        noisy: {
            train: dataNoisy.slice(0, trainSize),
            test: dataNoisy.slice(trainSize)
        }
    };

    console.log('Anzahl Trainingsdaten (clean):', result.clean.train.length);
    console.log('Anzahl Testdaten (clean):', result.clean.test.length);
    console.log('Anzahl Trainingsdaten (noisy):', result.noisy.train.length);
    console.log('Anzahl Testdaten (noisy):', result.noisy.test.length);

    return result;
}