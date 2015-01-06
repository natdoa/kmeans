package kmeans;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;

public class kmeans
{
	public static void main(String[] args)
	{ 
		System.out.println("Enter your number of clusters: ");
		Scanner scanner = new Scanner(System.in);
		int numClusters = Integer.parseInt(scanner.nextLine());
		scanner.close();

		System.out.println("\nBegin k-means clustering\n");

		File dataFile=new File("wine.data");
		double[][] rawData=readData(dataFile);
		System.out.println("Raw unclustered data:");
		System.out.println("-------------------");
		ShowData(rawData);

		System.out.println("\nSetting numClusters to " + numClusters);

		int[] clustering = Cluster(rawData, numClusters);

		System.out.println("\nK-means clustering for k="+ numClusters +" complete\n");

		System.out.println("Final clustering in vector form:\n");
		ShowVector(clustering);

		System.out.println("Raw data by cluster:\n");
		ShowClustered(rawData, clustering, numClusters);

		System.out.println("Error Rate:");
		int errorCount=ShowErrorCount(clustering, readIndices(dataFile));
		System.out.println("%"+((100*errorCount)/178)+"\n");

		System.out.println("Error Count:\n"+errorCount+"/178\n");


		double[][] means = GetClusterAverage(rawData, clustering, numClusters);
		System.out.println("Cluster center points:");
		ShowData(means);

		double avgRadius=GetAverageRadius(rawData, means, clustering, numClusters);
		System.out.println("\nAverage Radius for k= "+numClusters+" is "+ avgRadius+"\n");

		System.out.println("\nEnd k-means clustering\n");
	}

	// ============================================================================

	public static int[] Cluster(double[][] rawData, int numClusters)
	{
		double[][] data = Normalized(rawData);

		boolean changed = true;
		boolean success = true;

		int[] clustering = InitClustering(data.length, numClusters, 0);
		double[][] means = Allocate(numClusters, data[0].length);

		int maxCount = data.length * 10;
		int ct = 0;
		while (changed == true && success == true && ct < maxCount)
		{
			++ct; 
			success = UpdateMeans(data, clustering, means); 
			changed = UpdateClustering(data, clustering, means);
		}

		return clustering;
	}

	private static double[][] Normalized(double[][] rawData)
	{
		double[][] result = new double[rawData.length][];
		for (int i = 0; i < rawData.length; ++i)
		{
			result[i] = new double[rawData[i].length];
			System.arraycopy( rawData[i], 0, result[i], 0,rawData[i].length );        
		}

		for (int j = 0; j < result[0].length; ++j)
		{
			double colSum = 0.0;
			for (int i = 0; i < result.length; ++i)
				colSum += result[i][j];
			double mean = colSum / result.length;
			double sum = 0.0;
			for (int i = 0; i < result.length; ++i)
				sum += (result[i][j] - mean) * (result[i][j] - mean);
			double sd = sum / result.length;
			for (int i = 0; i < result.length; ++i)
				result[i][j] = (result[i][j] - mean) / sd;
		}
		return result;
	}

	private static int[] InitClustering(int numTuples, int numClusters, int randomSeed)
	{
		Random random = new Random(randomSeed);
		int[] clustering = new int[numTuples];
		for (int i = 0; i < numClusters; ++i) 
			clustering[i] = i;
		for (int i = numClusters; i < clustering.length; ++i)
			clustering[i] = random.nextInt(numClusters);

		return clustering;
	}

	private static double[][] Allocate(int numClusters, int numColumns)
	{
		double[][] result = new double[numClusters][];
		for (int k = 0; k < numClusters; ++k)
			result[k] = new double[numColumns];
		return result;
	}

	private static boolean UpdateMeans(double[][] data, int[] clustering, double[][] means)
	{
		int numClusters = means.length;
		int[] clusterCounts = new int[numClusters];
		for (int i = 0; i < data.length; ++i)
		{
			int cluster = clustering[i];
			++clusterCounts[cluster];
		}

		for (int k = 0; k < numClusters; ++k)
			if (clusterCounts[k] == 0)
				return false;

		for (int k = 0; k < means.length; ++k)
			for (int j = 0; j < means[k].length; ++j)
				means[k][j] = 0.0;

		for (int i = 0; i < data.length; ++i)
		{
			int cluster = clustering[i];
			for (int j = 0; j < data[i].length; ++j)
				means[cluster][j] += data[i][j];
		}

		for (int k = 0; k < means.length; ++k)
			for (int j = 0; j < means[k].length; ++j)
				means[k][j] /= clusterCounts[k];
		return true;
	}

	private static boolean UpdateClustering(double[][] data, int[] clustering, double[][] means)
	{
		int numClusters = means.length;
		boolean changed = false;

		int[] newClustering = new int[clustering.length];

		System.arraycopy( clustering, 0, newClustering, 0,clustering.length );

		double[] distances = new double[numClusters];

		for (int i = 0; i < data.length; ++i)
		{
			for (int k = 0; k < numClusters; ++k)
				distances[k] = Distance(data[i], means[k]);

			int newClusterID = MinIndex(distances);
			if (newClusterID != newClustering[i])
			{
				changed = true;
				newClustering[i] = newClusterID;
			}
		}

		if (changed == false)
			return false;

		int[] clusterCounts = new int[numClusters];
		for (int i = 0; i < data.length; ++i)
		{
			int cluster = newClustering[i];
			++clusterCounts[cluster];
		}

		for (int k = 0; k < numClusters; ++k)
			if (clusterCounts[k] == 0)
				return false;

		System.arraycopy( newClustering, 0, clustering, 0,newClustering.length );
		return true;
	}

	private static double Distance(double[] tuple, double[] mean)
	{
		double sumSquaredDiffs = 0.0;
		for (int j = 0; j < tuple.length; ++j){
			sumSquaredDiffs += Math.pow((tuple[j] - mean[j]), 2);
		}
		return Math.sqrt(sumSquaredDiffs);
	}

	private static int MinIndex(double[] distances)
	{
		int indexOfMin = 0;
		double smallDist = distances[0];
		for (int k = 0; k < distances.length; ++k)
		{
			if (distances[k] < smallDist)
			{
				smallDist = distances[k];
				indexOfMin = k;
			}
		}
		return indexOfMin;
	}

	static double[][] readData(File fIn){
		double[][] res= new double[178][];
		Path path = Paths.get(fIn.getPath());
		try (BufferedReader reader = Files.newBufferedReader(path)){
			String l = null;
			String[] line;
			double[] arr;
			for(int i=0;i<178;i++){
				l = reader.readLine();
				arr = new double[13];
				line = l.split(",");
				for(int j=1;j<14;j++){
					arr[j-1]=Double.parseDouble(line[j]);
				}
				res[i]=arr;
			}      
		} catch (IOException e) {
			System.err.println("Error reading data");
			System.exit(0);
		}
		return res;
	}

	static int[] readIndices(File fIn){
		int[] res= new int[178];
		Path path = Paths.get(fIn.getPath());
		try (BufferedReader reader = Files.newBufferedReader(path)){
			String l = null;
			String[] line;
			for(int i=0;i<178;i++){
				l = reader.readLine();
				line = l.split(",");
				res[i]=Integer.parseInt(line[0]);
			}      
		} catch (IOException e) {
			System.err.println("Error reading data");
			System.exit(0);
		}
		return res;
	}

	static double[][] GetClusterAverage(double[][] data, int[] clustering, int numClusters)
	{		
		double sumOfRadii=0;
		double countOfRadii=0;
		double[][] means=new double[numClusters][13];

		for (int j = 0; j < 13; ++j)
		{
			for (int k = 0; k < numClusters; ++k)
			{
				sumOfRadii=0;
				countOfRadii=0;
				for (int i = 0; i < data.length; ++i)
				{
					int clusterID = clustering[i];
					if (clusterID == k){
						sumOfRadii+=data[i][j];
						countOfRadii++;
					} 
				}
				means[k][j]=sumOfRadii/countOfRadii;
			}
		}
		return means;
	}

	static double GetAverageRadius(double[][] data, double[][] means, int[] clustering, int numClusters)
	{		
		double[] radii=new double[numClusters];
		double avg=0;

		for (int k = 0; k < numClusters; ++k)
		{
			double largestRadius=0;
			double dist=0;
			for (int i = 0; i < data.length; ++i)
			{
				int clusterID = clustering[i];
				if (clusterID == k){
					for(int j=0;j<data[i].length;j++){
						dist=Distance(data[i],means[k]);
						if(dist>largestRadius){
							largestRadius=dist;
						}
					}		
				} 
			}
			radii[k]=largestRadius;
		}
		
		for(int i=0;i<numClusters;i++){
			avg+=radii[i];
		}
		return avg/numClusters;
	}

	// ============================================================================

	static void ShowData(double[][] data)
	{
        DecimalFormat df = new DecimalFormat("#####.##");
		for (int i = 0; i < data.length; ++i)
		{
			for (int j = 0; j < data[i].length; ++j)
			{
				System.out.print(df.format(data[i][j]) + " ");
			}
			System.out.println("");
		}
		System.out.println("");
	}

	static void ShowVector(int[] vector)
	{
		for (int i = 0; i < vector.length; ++i)
			System.out.print((vector[i]+1) + " ");

		System.out.println("\n");
	}

	static int ShowErrorCount(int[] vector, int[] indices)
	{
		int errorCount=0;
		for (int i = 0; i < vector.length; ++i){
			if((vector[i]+1) != indices[i])
			{
				errorCount++;
			}
		}
		return errorCount; 
	}

	static void ShowClustered(double[][] data, int[] clustering, int numClusters)
	{
		for (int k = 0; k < numClusters; ++k)
		{
			System.out.println("===================");
			for (int i = 0; i < data.length; ++i)
			{
				int clusterID = clustering[i];
				if (clusterID != k) continue;
				System.out.print(i + " ");
				for (int j = 0; j < data[i].length; ++j)
				{
					if (data[i][j] >= 0.0) System.out.print(" ");
					System.out.print(data[i][j] + " ");
				}
				System.out.println("");
			}
			System.out.println("===================");
		}
	}
}