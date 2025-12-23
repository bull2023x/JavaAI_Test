# JavaAI_Test

# Java AI Inference PoC  
**ONNX Runtime + ResNet50 (ImageNet)**

## Overview

This project is a **Proof of Concept (PoC)** demonstrating that **modern AI inference can be implemented cleanly and reliably in Java**, without using Python at runtime.

The application performs **image classification** using a pre-trained **ResNet50** model (trained on ImageNet) via **ONNX Runtime**, and prints the **top-5 predictions** with confidence scores.

This PoC is designed to be:

- Minimal
- Reproducible
- Production-oriented
- Free from Python runtime dependencies

---

## What This Application Does

1. Accepts an image file (JPEG/PNG)
2. Preprocesses the image in Java:
   - Resize to 224×224
   - Normalize using ImageNet mean / standard deviation
3. Runs inference using **ONNX Runtime (native C++ engine)**
4. Applies softmax to output probabilities
5. Prints the **Top-5 ImageNet class predictions**

### Example Output



=== Top-5 Predictions ===

Labrador_retriever : 0.3786

golden_retriever : 0.2903

standard_poodle : 0.0759

otterhound : 0.0435

clumber : 0.0322


---

## Why This PoC Exists

Most AI examples today assume:
- Python
- Conda
- CUDA
- Jupyter notebooks

This project proves that:

- **Java is fully capable of running modern AI inference**
- ONNX Runtime provides production-grade performance
- AI can be embedded directly into JVM-based systems

This is especially useful for:
- Enterprise systems
- Backend services
- Existing Java platforms
- Regulated environments where Python is not preferred

---

## Technology Stack

- **Java 17**
- **Gradle**
- **ONNX Runtime (Java API)**
- **ResNet50 (ONNX format)**
- **ImageNet labels**

No Python runtime is required.

---

## Prerequisites

- macOS / Linux / Windows
- Java **17 or later**
- Internet connection (for first-time model download)

Verify Java:

```bash
java -version
```


Expected output (example):
```
java version "17.0.12"
```

Project Structure
```
onnx-java-clean/
├── README.md
├── app/
│   ├── build.gradle
│   ├── src/main/java/
│   │   └── Main.java
│   └── models/
│       ├── resnet50.onnx
│       └── imagenet_class_index.json
├── gradlew
├── settings.gradle
└── gradle/
```

Setup Instructions
Step 1️⃣ Clone or Create the Project

Create a working directory:
```
mkdir onnx-java-clean
cd onnx-java-clean
```

Step 2️⃣ Initialize a Java Application (Gradle)
```
gradle init --type java-application
```

Choose:

Language: Java

Build tool: Gradle

Test framework: JUnit (any is fine)

Step 3️⃣ Configure Gradle

Edit app/build.gradle:

```
plugins {
    id 'application'
}

repositories {
    mavenCentral()
}

dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime:1.17.1'
}

application {
    mainClass = 'Main'
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(17)
    }
}
```

Step 4️⃣ Download Model & Labels

Create the models directory:
```
mkdir -p app/models
```

Download ResNet50 (ONNX):
```
curl -L -o app/models/resnet50.onnx \
https://storage.googleapis.com/download.tensorflow.org/models/official/resnet/resnet50-v2-7.onnx
```

Download ImageNet labels:
```
curl -L -o app/models/imagenet_class_index.json \
https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
```

Verify:
```
ls -lh app/models
```
Step 5️⃣ Java Source Code

Create app/src/main/java/Main.java and paste the full Java implementation that:

Loads the ONNX model

Preprocesses the image

Runs inference

Applies softmax

Prints Top-5 predictions

(This PoC intentionally keeps everything in a single file for clarity.)

Running the Application

From the project root:
```
./gradlew run --args="/path/to/your/image.jpg"
```

Example:
```
./gradlew run --args="/Users/yourname/Downloads/dog.jpg"
```

Benefits of This Approach
✅ Java-Native AI Inference

No Python runtime

No Conda

No virtual environments

✅ Production-Ready

ONNX Runtime uses highly optimized native backends

Stable, versioned Java dependencies

✅ Easy Integration

Can be embedded into Spring Boot, Micronaut, or plain JVM services

Ideal for enterprise environments

Why Java for AI?

Strong typing and maintainability

Mature tooling and monitoring

Excellent performance with native libraries

Easier governance and security in enterprise systems

Java is not replacing Python for research —
it is complementing it for production.

License

This project is provided as a PoC and learning reference.
You may adapt and reuse it freely.

Final Notes

If you already have Java infrastructure, you do not need Python to deploy AI.
