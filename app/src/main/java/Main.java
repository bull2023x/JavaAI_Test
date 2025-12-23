import ai.onnxruntime.*;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.util.*;

public class Main {

    static final int SIZE = 224;

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: ./gradlew run --args=\"/path/to/image.jpg\"");
            System.exit(1);
        }

        File imageFile = new File(args[0]);
        File modelFile = new File("models/resnet50.onnx");
        File labelsFile = new File("models/imagenet_class_index.json");

        System.out.println("Model:  " + modelFile.getAbsolutePath());
        System.out.println("Labels: " + labelsFile.getAbsolutePath());
        System.out.println("Image:  " + imageFile.getAbsolutePath());

        BufferedImage img = ImageIO.read(imageFile);
        img = resize(img, SIZE, SIZE);

        float[] inputData = preprocess(img);

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession(modelFile.getPath());

        String inputName = session.getInputNames().iterator().next();

        OnnxTensor tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputData),
                new long[]{1, 3, SIZE, SIZE}
        );

        OrtSession.Result result = session.run(Map.of(inputName, tensor));
        float[] logits = ((float[][]) result.get(0).getValue())[0];
        float[] probs = softmax(logits);

        Map<Integer, String> labels = loadLabels(labelsFile);
        int[] top = topK(probs, 5);

        System.out.println("=== Top-5 Predictions ===");
        for (int i = 0; i < top.length; i++) {
            int idx = top[i];
            System.out.printf(
                    "%d) %s : %.4f%n",
                    i + 1,
                    labels.getOrDefault(idx, "class_" + idx),
                    probs[idx]
            );
        }
    }

    static BufferedImage resize(BufferedImage img, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.drawImage(img, 0, 0, w, h, null);
        g.dispose();
        return out;
    }

    static float[] preprocess(BufferedImage img) {
        float[] out = new float[3 * SIZE * SIZE];
        int r = 0, g = SIZE * SIZE, b = 2 * SIZE * SIZE;

        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                Color c = new Color(img.getRGB(x, y));
                out[r++] = (c.getRed() / 255f - 0.485f) / 0.229f;
                out[g++] = (c.getGreen() / 255f - 0.456f) / 0.224f;
                out[b++] = (c.getBlue() / 255f - 0.406f) / 0.225f;
            }
        }
        return out;
    }

static float[] softmax(float[] x) {
    float max = x[0];
    for (float v : x) {
        if (v > max) max = v;
    }

    float sum = 0f;
    float[] e = new float[x.length];
    for (int i = 0; i < x.length; i++) {
        e[i] = (float) Math.exp(x[i] - max);
        sum += e[i];
    }

    for (int i = 0; i < x.length; i++) {
        e[i] /= sum;
    }
    return e;
}

    static int[] topK(float[] probs, int k) {
        return java.util.stream.IntStream.range(0, probs.length)
                .boxed()
                .sorted((a, b) -> Float.compare(probs[b], probs[a]))
                .limit(k)
                .mapToInt(Integer::intValue)
                .toArray();
    }

    static Map<Integer, String> loadLabels(File file) throws Exception {
        String json = Files.readString(file.toPath());
        Map<Integer, String> map = new HashMap<>();
        String[] items = json.split("\"\\d+\":");
        int idx = 0;
        for (String item : items) {
            if (item.contains("[")) {
                String label = item.split("\"")[3];
                map.put(idx++, label);
            }
        }
        return map;
    }
}

