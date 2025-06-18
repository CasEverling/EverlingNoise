using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Drawing;
using System.Numerics;
using Microsoft.VisualBasic;

public class CENoiseMap
{

    #region HelperMethods
    internal static class NumberGenerator
    {
        public static float Next(float sd, float mean)
        {
            return Next(sd, mean, 0.001f);
        }

        public static float Next(float sd, float mean, float precision)
        {
            Random random = new Random();
            float x = (float)random.NextDouble();

            while (x < precision || x > 1 - precision)
                x = (float)random.NextDouble();

            return MathF.Log(x / (1 - x)) / 0.74036268949f * sd + mean;
        }
    }

    private static void Shuffle<T>(T[,] array, int length)
    {
        int width = array.Length / length, index;
        Random random = new Random();
        T[] temporary = new T[2];

        while (length > 1)
        {
            index = random.Next(length - 1);
            for (int i = 0; i <= width; i++)
            {
                temporary[i] = array[length - 1, i];
                array[length - 1, i] = array[index, i];
                array[index, i] = temporary[i];
            }
            length--;
        }
    }

    #endregion

    #region  Implementations

    //
    /
    public static float[,] StepMap(int[] size, float mean, float sd)
    {
        if (size.Length != 2)
            throw new ArgumentException("Size must contain exaclty two values");

        float[,] result = new float[size[0], size[1]];

        for (int i = 0; i < size[0]; i++)
            for (int j = 0; j < size[1]; j++)
                result[i, j] = NumberGenerator.Next(sd, mean);

        return result;
    }

    /*\
     *  RenderMap Fucntion Anlysis
     *  
     *  Time Complexity Variables: 
     *  -> Matrix size x = m
     *  -> Matrix size y = n
     *
     *  Time Complexity: 
     *  -> O(n * m)
     *
    \*/
    public static float[,] RenderMap(float[,] stepMap, int[] size, int[] startingPoint, float startingHeight)
    {
        // Using matrixes to check if nodes have been visted
        // Advanteges: access aways O(n)
        // Dictionary of HashSets would end up with the same size
        bool[,] renderMap = new bool[size[0], size[1]];
        bool[,] loadedMap = new bool[size[0], size[1]];
        int[,] adjacents = new int[8, 2];
        int adjSize;

        // Using fixed sized array for storing possible next values
        // Advatages: random acces always O(n)
        // Swapping variables: O(n)
        int[,] positionsToCheck = new int[size[0] * size[1] * 3 / 4 + 1, 2];
        positionsToCheck[0, 0] = startingPoint[0];
        positionsToCheck[0, 1] = startingPoint[1];
        int list_size = 1;

        int[] curr = startingPoint;
        int index;

        float sum; int len;

        Func<int, int, bool> isInsideMap = (x, y) =>
        {
            return !(x < 0 || y < 0 || x > size[0] || y > size[1]);
        };

        stepMap[startingPoint[0], startingPoint[1]] += startingHeight;

        do
        {
            /*\ CHOSING THE NEXT NODE
             *  
             *  This justifies using a matrix instead of a list
             *  Random access -> order does not matter
             *  Complexity of removing item in list = O(n)
             *  Complexity of swapping items in array = O(1)
             *
             *  Picking always the first or end nodes will create more organic shapes
             *  Picking random values will create a shape that resambles lines coming out of origin
             *
            \*/

            
            index = new Random().Next(list_size);
            curr[0] = positionsToCheck[index, 0];
            curr[1] = positionsToCheck[index, 1];

            // "Removing" item from list
            positionsToCheck[index, 0] = positionsToCheck[list_size - 1, 0];
            positionsToCheck[index, 1] = positionsToCheck[list_size - 1, 1];
            list_size--;


            // Adding adjacent coordinates to the list 
            sum = 0; len = 0; adjSize = 0;
            for (int i = curr[0] - 1; i <= curr[0] + 1; i++)
                for (int j = curr[1] - 1; j <= curr[1] + 1; j++)
                {
                    if (renderMap[i, j]) // Node already visited
                    {
                        // Adds to the average adjacent height
                        sum += stepMap[i, j];
                        len++;
                    }

                    else if (!loadedMap[i, j]) // Node not yet visited
                    {
                        // Decides what are possible next Steps
                        adjacents[adjSize, 0] = i;
                        adjacents[adjSize, 1] = j;
                        adjSize++;

                        // Confirming that node has been loaded to positionsToCheck
                        // This will be added later to the list,
                        // but is here because its close to conditional
                        loadedMap[i, j] = true;

                    }
                }

            // Randomizes Order of next Steps
            // Each position will only be featured on the adjacents list once, giving it perfonance of O(1n)
            Shuffle(adjacents, adjSize);
            for (int i = 0; i < adjSize; i++)
            {
                // Adding to the list
                positionsToCheck[list_size, 0] = adjacents[i, 0];
                positionsToCheck[list_size, 1] = adjacents[i, 1];
                list_size++;
            }

            // Averages the surroundings and add normal deviation
            stepMap[curr[0], curr[1]] += sum / len;
            renderMap[curr[0], curr[1]] = true;
            
        } while (list_size > 0);

        return stepMap;
    }

    #endregion
}