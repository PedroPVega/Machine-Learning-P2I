using System;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using L8 = SixLabors.ImageSharp.PixelFormats.L8;
using Unsupervised_Learning;

Simulation2 simulator = new Simulation2();

DeepLearner Torch = new DeepLearner(15,1);
Torch.InitializeDeepLearning();
Torch.Train(700, 10, 0.08);
//simulator.SimulateDeepLearning(1, 1);
// KMeans(10, minLoops:10,downloadImages : true);

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




