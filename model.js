function createModel() {
    // Erstelle ein sequentielles Modell
    console.log("Erstelle ein sequentielles Modell...");
    const model = tf.sequential();

    // 1. Input-Layer und 1. Hidden Layer mit 100 Neuronen und ReLU-Aktivierung
    console.log("Füge ersten Hidden Layer hinzu (Input-Layer + 100 Neuronen mit ReLU)... ");
    model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));

    // 2. Hidden Layer mit 100 Neuronen und ReLU-Aktivierung
    console.log("Füge zweiten Hidden Layer hinzu (100 Neuronen mit ReLU)...");
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));

    // 3. Output-Layer mit 1 Neuron und einer linearen Aktivierungsfunktion
    console.log("Füge Output-Layer hinzu (1 Neuron mit linearer Aktivierungsfunktion)...");
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    // Überprüfe die Modellzusammenfassung in der Konsole
    console.log("Modell erfolgreich erstellt. Zusammenfassung des Modells:");
    model.summary();

    return model;
}


function createOverfitModel() {
    const model = tf.sequential();

    // Mehr Schichten und Neuronen für höhere Komplexität
    // model.add(tf.layers.dense({ inputShape: [1], units: 64, activation: 'relu' }));
    // model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    // model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    // model.add(tf.layers.dense({ units: 1 })); // Output-Schicht (Regressionsmodell)

    model.add(tf.layers.dense({ inputShape: [1], units: 256, activation: 'relu' })); // größere Anzahl an Neuronen
    model.add(tf.layers.dense({ units: 256, activation: 'relu' })); // zusätzliche Hidden Layer
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 })); // Output Layer

    return model;
}