public class Simulation
{

    static double MaxDouble = double.MaxValue;
    public List<Subject> Subjects {get; set;}
    public List<Subject> Barycenters {get; set;}
    static Random rdn = new Random();
    public int[,] Mat{get; set;}
    public Simulation()
    {
        // Load Data into list
        Subjects = new List<Subject> {};
        Barycenters = new List<Subject> {};
        Mat = new int [0,0];
    }

    public void LoadData(int nbSubjects = 50)
    {
        int[] subjectsToLoad = new int[50]; // De 1 à 25, de 41 à 55, de 71 à 80
        for (int j = 0; j < 50; j++)
        {
            if (j < 25)
                subjectsToLoad[j] = j+1;
    
            else if (j<40)
                subjectsToLoad[j] = j+16;
            
            else
                subjectsToLoad[j] = j+31;
            
        }
        string tempName = "power_spectrum_sub_0";
        int t;
        for (int i = 0; i < subjectsToLoad.Length; i++)
        {
            t = subjectsToLoad[i];
            if (t < 10)
                Subjects.Add(new Subject(tempName + "0" + Convert.ToString(t) + ".csv"));
            
            else
                Subjects.Add(new Subject(tempName + Convert.ToString(t) + ".csv"));
        }
        Mat = new int[Subjects.Count,2];
    }

    public void CreateRandomBarycenters(int K) // Artificial point method
    {
        double[] minFreq = new double[10] {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
        double[] maxFreq = new double[10] {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
        double[] minPower = new double[10] {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
        double[] maxPower = new double[10] {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
        double temp;
        foreach (Subject sub in Subjects) 
        {
            for (int i = 0; i < 19; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int t = 0; t < 10; t++)
                    {
                        temp = sub.SubjectData[i,j,t];
                        if (j == 0)
                        {
                            if (minFreq[t] > temp)
                            {
                                minFreq[t] = temp;
                            } 
                            if (maxFreq[t] < temp)
                            {
                                maxFreq[t] = temp;
                            }
                        }
                        else
                        {
                            if (minPower[t] > temp)
                            {
                                minPower[t] = temp;
                            } 
                            if (maxPower[t] < temp)
                            {
                                maxPower[t] = temp;
                            }
                        }
                        
                    }
                }
            }
        }
        for (int k = 0; k < K; k++)
        {
            Barycenters.Add(new Subject(minFreq,maxFreq,minPower,maxPower));
        }
        
    }

    public void CreateRandomBaryv2(int K)
    {
        for (int k = 0; k < K; k++)
        {
            Barycenters.Add(new Subject(false));
        }
    }

    public void SelectSubjectsAsBarys(int K)
    {
        List<int> alreadySelected = new List<int>();
        int selected;
        for (int k = 0; k < K; k++)
        {
            selected = Convert.ToInt32(rdn.NextInt64(Subjects.Count));
            if (!alreadySelected.Contains(selected))
            {
                alreadySelected.Add(selected);
                Barycenters.Add(new Subject(Subjects[selected],false));
                Console.WriteLine("Barycenter selected : " + Subjects[selected].ToString());
            }
            else
                K--;
        }
    }

    public void SelectSubjectsKPlus(int K)
    {
        int selected;
        int[] nbSubsPerClass = new int[] {25,40,50};
        int lowerBound = 0;
        for (int k = 0; k < K; k++)
        {
            selected = Convert.ToInt32(rdn.NextInt64(lowerBound,nbSubsPerClass[k]));
            Barycenters.Add(new Subject(Subjects[selected],false));
            lowerBound = nbSubsPerClass[k];
            Console.WriteLine("Barycenter selected : " + Subjects[selected].ToString());
        }
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
        int i = 0;

        // foreach subject
        foreach (Subject sub in Subjects)
        {
            minDist = MaxDouble;
            for (int k = 0; k < K; k++)
            {
                // foreach subject we calculate its distance to all barycenters
                distances[k] = GetNorm2(Barycenters[k], sub);
                if(distances[k] < minDist)
                {
                    minDist = distances[k];
                    temp = k+1; // we assign its class to the nearest barycenter
                }
            }
            // assign nearest barycenter in matrix
            sub.DesignatedClass = temp;
            Mat[i,0] = sub.Id;
            Mat[i,1] = temp;
            i++;
        }
    }

    public void UpdateBarycenters(int K)
    {

        //Etape 1, remise à 0 des coordonnées des barycentres
        foreach (Subject bary in Barycenters)
        {
            bary.SetSubCoords2Zero(); 
        }

        //Etape 2, somme des coordonnées des points les plus proches à chaque barycentre
        int[] nbOfSubsPerClass = UpdatePart1(K);

        //Etape 3, diviser par le nombre de points afin de moyenner les coordonnées du barycentre
        foreach (Subject bary in Barycenters)
        {
            UpdatePart2(bary, nbOfSubsPerClass[- bary.Id - 1]);
        }
        //Console.WriteLine("nb de points groupe 1 : {0}, groupe 2 : {1}, groupe 3 : {2}, groupe 4 : {3}, groupe 5 : {4}",nbOfSubsPerClass[0],nbOfSubsPerClass[1],nbOfSubsPerClass[2],nbOfSubsPerClass[3],nbOfSubsPerClass[4]);
        //Console.WriteLine("nb de points groupe 1 : {0}, groupe 2 : {1}, groupe 3 : {2}",nbOfSubsPerClass[0],nbOfSubsPerClass[1],nbOfSubsPerClass[2]);
        Console.WriteLine("nb de points groupe 1 : {0}, groupe 2 : {1}",nbOfSubsPerClass[0],nbOfSubsPerClass[1]);
        // ShowClasses(K);
        //Etape 4, réinitialiser le barycentre s'il n'a aucun point attribué
        for (int i = 0; i < nbOfSubsPerClass.Length; i++)
        {
            if (nbOfSubsPerClass[i] == 0)
            {
                Barycenters.Insert(i, new Subject(Barycenters[i]));
                Barycenters.RemoveAt(i+1);
            }
        }
    }

    public double[,,] GetSubjectCoordsById(int id)
    {
        double[,,] coords = Subjects.FirstOrDefault(Subject => Subject.Id == id).SubjectData;
        if (coords != null)
        {
            return coords;
        }
        else
        {
            Subject s = new Subject();
            Console.WriteLine("SUJET PAS TROUVE, ERREUR");
            return s.SubjectData;
        }
    }

    public int[] UpdatePart1(int K)
    {
        int nbChannels = Barycenters[0].SubjectData.GetLength(0);
        int nbLocMaxima = Barycenters[0].SubjectData.GetLength(2);
        double[,,] temp;
        int[] nbOfSubsPerClass = new int[K];
        int closestBary;
        for (int k = 0; k < K; k++)
        {
            nbOfSubsPerClass[k] = 0;
        }

        for (int i = 0; i < Mat.GetLength(0); i++)
        {
            closestBary = Mat[i,1];
            nbOfSubsPerClass[closestBary-1] += 1;

            temp = GetSubjectCoordsById(Mat[i,0]);
            for (int j = 0; j < nbChannels; j++)
            {
                for (int t = 0; t < nbLocMaxima; t++)
                {
                    Barycenters[closestBary - 1].SubjectData[j,0,t] += temp[j,0,t];
                    Barycenters[closestBary - 1].SubjectData[j,1,t] += temp[j,1,t];
                }
            }
        }
        int s = 0;
        foreach (int item in nbOfSubsPerClass)
        {
            s += item;
        }
        if (!(s == 50))
        {
            Console.WriteLine("ERROR");
        }
        return nbOfSubsPerClass;
    }

    public void UpdatePart2(Subject barycenter, int n)
    {
        for (int i = 0; i < barycenter.SubjectData.GetLength(0); i++)
        {
            for (int j = 0; j < barycenter.SubjectData.GetLength(2); j++)
            {
                barycenter.SubjectData[i,0,j] /= n;
                barycenter.SubjectData[i,1,j] /= n;
            }
        }
    }

    public void InitializeKMeans() // All subject with no class
    {
        for (int i = 0; i < Subjects.Count; i++)
        {
            Mat[i,0] = Subjects[i].Id;
            Mat[i,1] = 0; // Classe initiale désignée à zero
        }
    }

    public double GetNorm1(Subject subCentre, Subject subPeripherie)
    {
        double distance = 0;
        for (int i = 0; i < 19; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int t = 0; t < 10; t++)
                {
                    distance += Math.Abs(subCentre.SubjectData[i,j,t] - subPeripherie.SubjectData[i,j,t]);
                }
            }
        }
        return distance;
    }

    public double GetNorm2(Subject subCentre, Subject subPeripherie)
    {
        double distance = 0;
        double temp;
        for (int i = 0; i < 19; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int t = 0; t < 10; t++)
                {
                    temp = subCentre.SubjectData[i,j,t] - subPeripherie.SubjectData[i,j,t];
                    distance += temp*temp;
                }
            }
        }
        return Math.Pow(distance, 0.5);
    }

    public void ShowBarycenters()
    {
        foreach (Subject bary in Barycenters)
        {
            Console.WriteLine("Coordonnées du barycentre {0}", -bary.Id);
            bary.PrintCoords();
        }
    }

    public void Simulate(int K, int loops)
    {
        this.LoadData();
        SelectSubjectsAsBarys(K);
        this.InitializeKMeans();
        for (int i = 0; i < loops; i++)
        {
            Console.WriteLine();
            Console.WriteLine("Boucle numéro : {0}", i+1);
            this.KmeansIteration(K);
            //ShowBarycenters();
            //Console.WriteLine("Distance between barycenters : {0}", GetNorm1(Barycenters[0],Barycenters[1]));
        }
        
    }

    public void ShowClasses(int K)
    {
        for (int k = 1; k < K+1; k++)
        {
            Console.WriteLine("Classe {0} :",k);
            foreach(Subject sub in Subjects)
            {
                if (sub.DesignatedClass == k)
                {
                    Console.WriteLine(sub.ToString());
                }
            }
            Console.WriteLine("-------------------------------");
        }
    }
}