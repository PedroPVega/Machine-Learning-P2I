using System.ComponentModel;
using System.Reflection.Emit;

public class Simulation2 
{
    static double MaxDouble = double.MaxValue;
    public static int ImgSize = 784;
    public List<Number> Numbers {get; set;}
    public List<Number> Barycenters {get; set;}
    static Random rdn = new Random();
    public Simulation2()
    {
        // Load Data into list
        Numbers = new List<Number> {};
        Barycenters = new List<Number> {};
        Console.WriteLine("Loading data");
        LoadData();
        Console.Clear();
    }

    public void SimulateDeepLearning(int l, int L)
    {
        /*
        l : number of hidden layers
        L : lenght of hidden layers
        */

        // DECLARING VARIABLES
        double[][] hiddenLayers = new double[L][];
        double[] finalLabels = new double[10];
        double cost;
        List<double[,]> Weights = new List<double[,]>();
        List<double[,]> DeltaWeights = new List<double[,]>();
        List<double[]> Biases = new List<double[]>();
        List<double[]> DeltaBiases = new List<double[]>();

        // FILL MATRIXES
        InitializeDeepLearning(l,L,hiddenLayers,Weights,Biases, DeltaWeights, DeltaBiases);

        
        // CHOOSE EXAMPLE
        Number example = Numbers[0];
        Console.WriteLine("this number is a {0}",example.Label);

        // PREDICT EXAMPLE
        
        double[] vect = new double[ImgSize];
        for (int i = 0; i < ImgSize; i++)
        {
            vect[i] = example.Pixels[i]/255d;        
        }
        //Console.WriteLine("integers converted to doubles");
        finalLabels = Predict(vect,hiddenLayers,Weights,Biases);
        cost = GetError(example.Label,finalLabels);
        //PrintArray(finalLabels);
        //Console.WriteLine("Erreur de la prediction : {0}",GetError(example.Label,finalLabels));
        
        /*
        // AVERAGE COST
        double avrCost = GetAverageCost(hiddenLayers,Weights,Biases);
        Console.WriteLine("Coût moyen des predictions : {0}",avrCost);
        */

        // WEIGHTS FOR 1 NUMBER 1 LAYER OF 1 NEURON
        double[,] dw1 = DeltaWeights[1];
        double[,] w1 = Weights[1];
        double[] b1 = Biases[1];
        double[] db1 = DeltaBiases[1];
        double temp;
        // Console.WriteLine("nombre de matrices de poids {0} ; nombre de vecteurs biais {1}", Weights.Count, Biases.Count);
        for (int i = 0; i < finalLabels.Length; i++)
        {
            if (i == example.Label)
            {
                temp = 2 * (finalLabels[i] - 1) * (d_dx_Sigmoid(w1[0,0]) * hiddenLayers[0][0] + b1[0]);
                dw1[0,i] = temp * hiddenLayers[0][0];
                db1[i] = temp;
            }
            else
            {
                temp = 2 * (finalLabels[i] - 0) * (d_dx_Sigmoid(w1[0,0]) * hiddenLayers[0][0] + b1[0]);
                dw1[0,i] = temp * hiddenLayers[0][0];
                db1[i] = temp;
            }
            Console.WriteLine("dw1[{0},1] = {1}    ; db1[{0}] = {2} ",i,Math.Round(dw1[0,i],2),Math.Round(db1[i],2));
        }
        
    }

    public void SimulateKMeans(int K, int iterations)
    {
        CreateRandomBarycenters(K);
        for (int i = 0; i < iterations; i++)
        {
            // Console.WriteLine("Itération {0} en cours", i+1);
            KmeansIteration(K);
        }
    }

    public void Reset()
    {
        foreach (Number item in Numbers)
        {
            item.DesignatedClass = -1;
        }
        Barycenters.Clear();
        Numbers[0].ResetBarsIds();
    }

    public void KmeansIteration(int K)
    {
        UpdateClasses(K);
        UpdateBarycenters(K);
    }

    public void UpdateClasses(int K)
    {
        double[] distances = new double[K];
        double minDist = MaxDouble;
        int temp = 0;

        // foreach point
        foreach (Number num in Numbers)
        {
            minDist = MaxDouble;
            for (int k = 0; k < K; k++)
            {
                // foreach point we calculate its distance to all barycenters
                distances[k] = GetDist1(Barycenters[k], num);
                if(distances[k] < minDist)
                {
                    minDist = distances[k];
                    temp = k; // we assign its class to the nearest barycenter
                }
            }
            // assign nearest barycenter in matrix
            num.DesignatedClass = temp;
        }
    }

    public void UpdateBarycenters(int K)
    {

        //Etape 1, remise à 0 des coordonnées des barycentres
        foreach(Number bary in Barycenters)
            bary.Set2Empty(); 

        //Etape 2, somme des coordonnées des points les plus proches à chaque barycentre
        int[] nbOfSubsPerClass = UpdatePart1(K);
        //PrintPointsPerClass(nbOfSubsPerClass);

        //Etape 3, diviser par le nombre de points afin de moyenner les coordonnées du barycentre
        foreach (Number bary in Barycenters)
        {
            Console.WriteLine("La creature {0}", -bary.Id);
            UpdatePart2(bary, nbOfSubsPerClass[- bary.Id]);
        }

        //Etape 4, en cas d'une classe vide, on cherche à reinitialiser le barycentre
        if (nbOfSubsPerClass.Contains(0))
        {
            int bar = 0;
            Number num;
            for (int i = 0; i < nbOfSubsPerClass.Length; i++)
            {
                if (nbOfSubsPerClass[i] == 0)
                {
                    bar = i;
                    break;
                }
            }
            num = Numbers[rdn.Next(Numbers.Count)];
            //Console.WriteLine("oupigoupi parmis {0}, je cherche {1}", num.DesignatedClass, bar);
            Barycenters[bar] = new Number(num);
        }
    }

    public int[] UpdatePart1(int K)
    {
        int[] nbOfSubsPerClass = new int[K];
        FillIntZeros(nbOfSubsPerClass);

        foreach (Number num in Numbers)
        {
            //Console.WriteLine("{0}",num.DesignatedClass);
            nbOfSubsPerClass[num.DesignatedClass] += 1;
            for (int i = 0; i < 784; i++)
                Barycenters[num.DesignatedClass].Pixels[i] += num.Pixels[i];
            
        } 

        int s = 0;
        foreach (int item in nbOfSubsPerClass)
            s += item;

        if (s != 12000)
            Console.WriteLine("ERROR");

        return nbOfSubsPerClass;
    }

    public void PrintPointsPerClass(int[] nbOfSubsPerClass)
    {
        
        double[] meanLabel = new double[nbOfSubsPerClass.Length];
        FillDoubleZeros(meanLabel);
        int p;
        foreach (Number item in Numbers)
        {
            p = item.DesignatedClass;
            meanLabel[p] += item.Label;
        }
        for (int i = 0; i < nbOfSubsPerClass.Length; i++)
        {
            meanLabel[i] = meanLabel[i] / nbOfSubsPerClass[i];
        }
        for(int i = 0; i < Barycenters.Count; i++)
        {
            Console.WriteLine("Points du barycentre {0} : {1} points attribués, label moyen : {2}", Barycenters[i].Id, nbOfSubsPerClass[i], Math.Round(meanLabel[i],2));
        }
    }

    public void UpdatePart2(Number barycenter, int n)
    {
        if(n != 0)
        {
            for (int i = 0; i < barycenter.Pixels.Length; i++)
            {
                barycenter.Pixels[i] /= n;
            }
        }
        else
        {
            Number c = new Number(null, false, false, true);
            for (int i = 0; i < barycenter.Pixels.Length; i++)
            {
                barycenter.Pixels[i] = c.Pixels[i];
            }
        }
    }

    public void LoadData()
    {
        string[]? values;
        int[] vector = new int[785];
        string filePath = "../mnist_train.csv";
        using (StreamReader reader = new StreamReader(filePath))
        {
            string? line = reader.ReadLine(); // Ignore first line, titles
            for (int i = 0; i < 12000; i++) // Load first 12000 numbers
            {
                
                    line = reader.ReadLine();  // Read line
                    if (line != null)
                    {
                        values = line.Split(','); // Split by comma
                        for (int t = 0; t < values.Length; t++) 
                            vector[t] = Convert.ToInt32(values[t]);                        
                
                    }
                Numbers.Add(new Number(vector));
            } 
        }
    }

    private void CreateRandomBarycenters(int K)
    {
        for (int k = 0; k < K; k++)
        {
            Barycenters.Add(new Number(null, false, true, true));
        }
    }

    public void SelectRandomBarycenters(int K)
    {
        int selected;
        List<int> alreadySelected = new List<int>();
        for (int k = 0; k < K; k++)
        {
            selected = Convert.ToInt32(rdn.NextInt64(0,Numbers.Count));
            if (!alreadySelected.Contains(selected))
            {
                Barycenters.Add(new Number(Numbers[selected].Pixels,false,true,false));
                //Console.WriteLine("Barycenter selected : " + Numbers[selected].ToString());
                alreadySelected.Add(selected);
            }
            else
                k--;
 
        }
    }

    public void SelectRandomBarycentersPlus(int K) //KMeans++
    {
        int selected;
        int selectedLabel;
        List<int> alreadySelected = new List<int>();
        for (int k = 0; k < K; k++)
        {
            selected = Convert.ToInt32(rdn.NextInt64(0,Numbers.Count));
            selectedLabel = Numbers[selected].Label;
            if (!alreadySelected.Contains(selectedLabel))
            {
                Barycenters.Add(new Number(Numbers[selected].Pixels,false,true,false));
                //Console.WriteLine("Barycenter selected : " + Numbers[selected].ToString());
                alreadySelected.Add(selectedLabel);
            }
            else
                k--;
 
        }
    }

    public double GetDist1(Number point, Number center)
    {
        double distance = 0;
        for (int t = 0; t < point.Pixels.Length; t++)
        {
            distance += Math.Abs(point.Pixels[t] - center.Pixels[t]);
        }
        return distance;
    }

    public double GetDist2(Number point, Number center)
    {
        double distance = 0;
        double temp;

        for (int t = 0; t < point.Pixels.Length; t++)
        {
            temp = point.Pixels[t] - center.Pixels[t];
            distance += temp*temp;
        }
        return Math.Pow(distance, 0.5);
    }

    public void ShowClasses(int K)
    {
        int[] nbPtsPerClass = new int[K];
        FillIntZeros(nbPtsPerClass);
        foreach (Number item in Numbers)
        {
            nbPtsPerClass[item.DesignatedClass] += 1;
        }
        PrintPointsPerClass(nbPtsPerClass);
    }

    public double GetAccuracy()
    {
        int K = Barycenters.Count;
        int temp;
        double acc = 0;
        double classAcc;
        int[,] mat = new int[K, 11]; // 10 digits + le total de points pour la classe
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < 11; j++)
            {
                mat[i,j] = 0;
            }
        }
        // Chaque ligne correspond à un barycentre
        // 1e colonne : taille du cluster
        // 2e à K+1-ième colonne : nombre d'apparitions de chaque label dans ce cluster
        
        // pour chaque classe
        foreach (Number num in Numbers)
        {
            mat[num.DesignatedClass,0] += 1; // determiner la longueur de la classe
            mat[num.DesignatedClass,num.Label+1] += 1; 
        }

        for (int i = 0; i < K; i++)
        {
            temp = -1;
            for (int j = 1; j < 11; j++)
            {
                if (mat[i,j] > temp)
                {
                    temp = mat[i,j]; // determiner le label le plus commun de chaque classe
                }
            }
            if (mat[i,0] != 0)
            {
                classAcc = (double)temp / (double)mat[i,0];
                acc += classAcc; // calculer la proportion d'apparition de se label dans cette classe
            }
        }
        acc = acc / K;
        return acc * 100;
        // calculer la moyenne des proportions
        // renvoyer la moyenne pondéré
        ///////////////////////////////////////
        
    }

    public void FillIntZeros(int[] array)
    {
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = 0;
        }
    }

    public void FillDoubleZeros(double[] array)
    {
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = 0;
        }
    }

    public void FillRandomblyMatrix2(double[][] mat)
    {
        for (int i = 0; i < mat.Length; i++)
        {
            for (int j = 0; j < mat[i].Length; j++)
            {
                mat[i][j] = rdn.NextDouble();
            }
        }
    }

    public void FillRandomblyMatrix(double[,] mat)
    {
        for (int i = 0; i < mat.GetLength(0); i++)
        {
            for (int j = 0; j < mat.GetLength(1); j++)
            {
                mat[i,j] = rdn.NextDouble();
            }
        }
    }

    public void FillRandomblyVector(double[] vect)
    {
        for (int i = 0; i < vect.Length; i++)
        {
            vect[i] = rdn.NextDouble();
        }
    }

    public void InitializeDeepLearning(int l, int L, double[][] HL, List<double[,]> W, List<double[]> B, List<double[,]> dW, List<double[]> dB)
    {
        Console.WriteLine("Initializing matrixes and vectors");
        for (int i = 0; i < HL.GetLength(0); i++)
        {
            HL[i] = new double[l];
        }
        FillRandomblyMatrix2(HL);
        W.Add(new double[ImgSize,L]);
        dW.Add(new double[ImgSize,L]);
        B.Add(new double[L]);
        dB.Add(new double[L]);
        FillRandomblyVector(B[0]);
        for (int i = 1; i < l; i++)
        {
            W.Add(new double[L,L]);
            dW.Add(new double[L,L]);
            FillRandomblyMatrix(W[i]);
            B.Add(new double[L]);
            dB.Add(new double[L]);
            FillRandomblyVector(B[i]);
        }
        W.Add(new double[L,10]);
        dW.Add(new double[L,10]);
        FillRandomblyMatrix(W[l]);
        B.Add(new double[10]);
        dB.Add(new double[10]);
        FillRandomblyVector(B[l]);
        Console.Clear();
        Console.WriteLine("Neural Network Initialized !!");
    }

    public double[] Predict(double[] vector, double[][] HL, List<double[,]> W, List<double[]> B)
    {
        // DECLARING VARIABLES
        double[,] wi; // weight i
        double[] bi; // biais i
        double[] prediction = new double[10];
        int L = HL.Length; // depth of hidden layer
        int l = HL[0].Length; // number of hidden layers

        // FIRST COLUMN
        wi = W[0];
        bi = B[0];
        //Console.WriteLine("Starting layer 0");
        for (int h = 0; h < L; h++)
        {
            for (int j = 0; j < ImgSize; j++)
            {
                HL[h][0] += wi[j,h] * vector[j];
            }
            //PrintArray(bi);
            HL[h][0] = Sigmoid(HL[h][0] + bi[h]);
        }
        //PrintJaggedArray(HL);
        //Console.WriteLine("Layer 0 done with sigmoid");

        // SECOND COLUMN THROUGH l COLUMN
        for (int i = 1; i < l; i++) 
        {
            //Console.WriteLine("Starting layer {0}",i);
            // FOREACH HIDDEN LAYER
            wi = W[i];
            bi = B[i];
            for (int h = 0; h < L; h++)
            {
                for (int j = 0; j < L; j++)
                {
                    HL[h][i] += wi[j,h] * HL[h][i-1];
                }
                //PrintArray(bi);
                HL[h][i] = Sigmoid(HL[h][i] + bi[h]);
            }
            //Console.WriteLine("Layer {0} done without sigmoid",i);
            //PrintJaggedArray(HL);
            //Console.WriteLine("Layer {0} done with sigmoid",i);
        }

        // LAST COLUMN
        wi = W[W.Count-1];
        bi = B[B.Count-1];
        //Console.WriteLine("Starting last layer");
        for (int h = 0; h < prediction.Length; h++)
        {
            for (int j = 0; j < L; j++)
            {
                prediction[h] += wi[j,h] * HL[j][HL[j].Length-1];
            }
            //PrintArray(bi);
            prediction[h] = Sigmoid(prediction[h] + bi[h]);
        }
        //PrintJaggedArray(HL);
        //Console.WriteLine("Last layer done with sigmoid");
        return prediction;
    }

    public double GetError(int label, double[] prediction)
    {
        double error = 0;
        for (int i = 0; i < prediction.Length; i++)
        {
            if (i == label)
            {
                error += Math.Pow(1-prediction[i],2);
            }
            else
            {
                error += Math.Pow(prediction[i],2);
            }
        }
        return error;
    }

    public double GetAverageCost(double[][] HL, List<double[,]> W, List<double[]> B)
    {
        double sumCost = 0;
        double[] vect = new double[ImgSize];
        foreach (Number num in Numbers)
        {
            for (int i = 0; i < ImgSize; i++)
            {
                vect[i] = num.Pixels[i]/255d;        
            }
            //PrintArray(finalLabels);
            sumCost += GetError(num.Label,Predict(vect,HL,W,B));
        }
        return sumCost / Numbers.Count;
    }

    public void BackPropagation()
    {

    }

    public double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public double d_dx_Sigmoid(double x)
    {
        double term = Math.Exp(-x);
        return term / Math.Pow(term + 1,2);
    }

    static void PrintJaggedArray(double[][] jaggedArray)
    {
        for (int i = 0; i < jaggedArray.Length; i++)
        {
            Console.Write("Row " + i + ": ");
            foreach (double num in jaggedArray[i])
            {
                Console.Write(Math.Round(num,2) + " ");
            }
            Console.WriteLine();
        }
    }

    static void PrintArray(double[] jaggedArray)
    {
        
        Console.Write("Array : ");
        foreach (double num in jaggedArray)
        {
            Console.Write(Math.Round(num,2) + " ");
        }
        Console.WriteLine();
        
    }
}