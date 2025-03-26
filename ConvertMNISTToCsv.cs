using System;
using System.IO;
using System.Text;

class ConvertMNISTToCSV
{
    private int ReverseBytes(int value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    private void ConvertMNIST(string imagesPath, string labelsPath, string outputPath)
    {
        using (BinaryReader images = new BinaryReader(File.Open(imagesPath, FileMode.Open)))
        using (BinaryReader labels = new BinaryReader(File.Open(labelsPath, FileMode.Open)))
        using (StreamWriter outputFile = new StreamWriter(outputPath))
        {
            int magicNumberImages = ReverseBytes(images.ReadInt32());
            int numImages = ReverseBytes(images.ReadInt32());
            int numRows = ReverseBytes(images.ReadInt32());
            int numCols = ReverseBytes(images.ReadInt32());

            int magicNumberLabels = ReverseBytes(labels.ReadInt32());
            int numLabels = ReverseBytes(labels.ReadInt32());

            if (numImages != numLabels)
            {
                Console.WriteLine("Number of images and labels do not match!");
                return;
            }

            Console.WriteLine($"Converting {numImages} images...");

            // Write CSV header (label + 784 pixels)
            outputFile.WriteLine("label," + string.Join(",", new string[784].Select((_, i) => "pixel" + i)));

            for (int i = 0; i < numImages; i++)
            {
                byte label = labels.ReadByte();
                byte[] pixels = images.ReadBytes(numRows * numCols);

                string csvLine = label + "," + string.Join(",", pixels);
                outputFile.WriteLine(csvLine);
            }

            Console.WriteLine($"Conversion completed! CSV saved to {outputPath}");
        }
    }

    public void Main()
    {
        string trainImages = "./archive/train-images.idx3-ubyte";
        string trainLabels = "./archive/train-labels.idx1-ubyte";
        string outputCSV = "mnist_train.csv";

        ConvertMNIST(trainImages, trainLabels, outputCSV);
    }
}
