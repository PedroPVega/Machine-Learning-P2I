using System.ComponentModel;
using System.Reflection.Emit;

public class Simulation2 
{
    static double MaxDouble = double.MaxValue;
    public static int ImgSize = 784;
    public List<Number> Numbers {get; set;}
    public List<Number> Barycenters {get; set;}
    protected Random rdn = new Random();
    public Simulation2()
    {
        // Load Data into list
        Numbers = new List<Number> {};
        Barycenters = new List<Number> {};
        Console.WriteLine("Loading data");
        LoadData();
        Console.Clear();
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