
using System;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using L8 = SixLabors.ImageSharp.PixelFormats.L8;
using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Versatile;
using Encog.ML.Train;
using Encog.Neural.Networks.Training.Propagation.Back;
using Unsupervised_Learning;

Simulation2 simulator = new Simulation2();

//var model = new Sequential();
//model.Add(new Dense(32, activation: "relu", input_shape: new Shape(2)));
/*
var network = new BasicNetwork();
network.AddLayer(new BasicLayer(null, true, 784));       // Input layer
network.AddLayer(new BasicLayer(new ActivationReLU(), true, 64));  // Hidden layer
network.AddLayer(new BasicLayer(new ActivationSoftMax(), false, 10)); // Output
network.Structure.FinalizeStructure();
network.Reset();
*/
const int InputSize = 784;
const int OutputSize = 10;
const int H = 32; // Neurons per hidden layer
const int L = 2;  // Number of hidden layers

string path = "../mnist_train.csv"; // Path to CSV file
var (data, labels) = LoadCsv(path, 7000);

int total = data.Count;
int trainCount = total - 1000;

var trainingSet = new BasicMLDataSet();

for (int i = 0; i < trainCount; i++)
{
    trainingSet.Add(new BasicMLDataPair(data[i], labels[i]));
}

var testInputs = data.Skip(trainCount).ToArray();
var testOutputs = labels.Skip(trainCount).ToArray();

var network = CreateNetwork();

var train = new Backpropagation(network, trainingSet);
train.NumThreads = 1;
for (int epoch = 0; epoch < 30; epoch++)
{
    train.Iteration();
    Console.WriteLine($"Epoch {epoch + 1}, Error: {train.Error:F4}");
}

// Evaluate on last 1000 samples
int correct = 0;
for (int i = 0; i < testInputs.Length; i++)
{
    var output = network.Compute(testInputs[i]);
int predicted = ArgMax(output);  // No need for .Data here
int actual = ArgMax(testOutputs[i]);

    if (predicted == actual) 
        correct++;
}

double accuracy = (double)correct / testInputs.Length * 100;
Console.WriteLine($"Test Accuracy: {accuracy:F2}%");

//EncogFramework.Instance.Shutdown();
    

    static BasicNetwork CreateNetwork()
    {
        var network = new BasicNetwork();
        network.AddLayer(new BasicLayer(null, true, InputSize));

        for (int i = 0; i < L; i++)
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, H));

        network.AddLayer(new BasicLayer(new ActivationSoftMax(), false, OutputSize));

        network.Structure.FinalizeStructure();
        network.Reset();
        return network;
    }

    static (List<IMLData> inputs, List<IMLData> outputs) LoadCsv(string file, int a)
    {
        var inputs = new List<IMLData>();
        var outputs = new List<IMLData>();
        var lines = File.ReadAllLines("../mnist_train.csv").Skip(1); // Skip the header
        foreach (var line in lines)
        {
            a--;
            var parts = line.Split(',').Select(double.Parse).ToArray();
            var label = (int)parts[0];
            for (int i = 0; i < parts.Length; i++)
            {
                parts[i] = parts[i]/255.0;
            }
            var input = new BasicMLData(parts.Skip(1).ToArray());

            // one-hot encoding
            var outputArr = new double[10];
            outputArr[label] = 1.0;
            var output = new BasicMLData(outputArr);

            inputs.Add(input);
            outputs.Add(output);
            if (a == 0)
                return (inputs, outputs);
        }

        return (inputs, outputs);
    }

static int ArgMax(IMLData data)
{
    var basicData = (BasicMLData)data;  // Cast to BasicMLData
    var values = basicData.Data;        // Access the Data property
    return values.ToList().IndexOf(values.Max());  // Find the index of the maximum value
}

/*
DeepLearner Torch = new DeepLearner(128,1,0.01);
Torch.InitializeDeepLearning();
Torch.Fit(Torch.XTrain_YTrain, 100, 16);
// simulator.SimulateDeepLearning(1, 1);
// KMeans(10, minLoops:10,downloadImages : true);

Simulation2 Km = new Simulation2();

List<Number> liste = new List<Number>();
string title;
for (int i = 0; i < 20; i++)
{
    liste.Add(Km.Numbers[i]);
    title = "chiffre ";
    SaveImage(liste[i].Pixels, title + Convert.ToString(i));
}
*/
/*
var allData = NeuralNetwork.LoadMNIST("../mnist_train.csv", maxSamples: 7000); // or however many you have
NeuralNetwork net = new NeuralNetwork(hiddenSize: 128, numHiddenLayers: 2);
// Load all data


        // Split data
        var trainData = allData.Take(allData.Count - 1000).ToList();
        var testData = allData.Skip(allData.Count - 1000).ToList();


        // Train
        for (int epoch = 0; epoch < 30; epoch++)
        {
            foreach (var (input, label) in trainData)
                net.Backward(input, label);

            Console.WriteLine($"Epoch {epoch + 1} complete.");
        }

        // Validate/Test
        int correct = 0;
        foreach (var (input, label) in testData)
        {
            var (activations, _) = net.Forward(input);
            int predicted = ArgMax(activations.Last());
            if (predicted == label)
                correct++;
        }

        double accuracy = correct / 10.0;
        Console.WriteLine($"Test Accuracy: {accuracy:F2}%");
    

static int ArgMax(double[] vector)
{
    int maxIdx = 0;
    double maxVal = vector[0];
    for (int i = 1; i < vector.Length; i++)
    {
        if (vector[i] > maxVal)
        {
            maxVal = vector[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}
*/
void KMeans(int minK, int upperK = -1,int minLoops = 1, int testCount = 1,bool downloadImages = false)
{
    string title;
    double meanAcc = 0;
    if (upperK == -1)
        upperK = minK + 1;
    for (int k = minK; k < upperK; k++)
    {

        Console.WriteLine("Tests pour classification à {0} clusters",k);
        for (int i = 0; i < testCount; i++) // 10 tests for each value of K
        {
            Console.WriteLine("{0} % done",i*10);
            simulator.SimulateKMeans(k,minLoops);
            meanAcc+=simulator.GetAccuracy();
            if (i != testCount-1)
                simulator.Reset();
            Console.SetCursorPosition(0, Console.CursorTop - 1);
            ClearCurrentConsoleLine();
        }
        
        meanAcc /= 10d;
        Console.WriteLine("Mean accuracy : {0} %",meanAcc);
        Console.WriteLine("-------------------------------------------------------");
        if (downloadImages)
        {
            Console.WriteLine("coucou");
            foreach (Number item in simulator.Barycenters)
            {
                title = "Barycentre " + Convert.ToString(-item.Id);
                SaveImage(item.Pixels, title);
            }
        }
    }
}

// Create the image
void SaveImage(int[] vector, string title)
{
    using (var image = new Image<L8>(28, 28))
    {
    // Fill the image with pixel values
        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                image[x, y] = new L8(Convert.ToByte(vector[28*y+x]));
            }
        }
        string filePath = title+".png";
        image.Save(filePath);
        Console.WriteLine($"Image saved as {filePath}");
    }
}

static void ClearCurrentConsoleLine()
{
    int currentLineCursor = Console.CursorTop;
    Console.SetCursorPosition(0, Console.CursorTop);
    Console.Write(new string(' ', Console.WindowWidth)); 
    Console.SetCursorPosition(0, currentLineCursor);
}