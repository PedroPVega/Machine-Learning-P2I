using System;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using L8 = SixLabors.ImageSharp.PixelFormats.L8;
// Simulation sim1 = new Simulation();
// sim1.Simulate(3,30);

Simulation2 simulator = new Simulation2();

double meanAcc = 0;
for (int k = 10; k < 11; k++)
{

    Console.WriteLine("Tests pour classification à {0} clusters",k);
    for (int i = 0; i < 1; i++) // 10 tests for each value of K
    {
        Console.WriteLine("{0} % done",i*10);
        simulator.Simulate(k,10);
        meanAcc+=simulator.GetAccuracy();
        //simulator.Reset();
        Console.SetCursorPosition(0, Console.CursorTop - 1);
        ClearCurrentConsoleLine();
    }
    meanAcc /= 10d;
    Console.WriteLine("Mean accuracy : {0} %",meanAcc);
    Console.WriteLine("-------------------------------------------------------");
    string title;
    foreach (Number item in simulator.Barycenters)
    {
        title = "Barycentre " + Convert.ToString(-item.Id);
        SaveImage(item.Pixels, title);
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




