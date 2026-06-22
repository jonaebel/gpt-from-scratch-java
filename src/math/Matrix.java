package math;

public class Matrix {

    double[][] values;

    /*
        Initilizes a (double) Matrix of size n x m with the value 0
            @param n rows
            @param m colums
     */
    public Matrix(int n, int m) {
        values = new double[n][m];
    }

    /*
        Initlizes a (double) Matrix with values
            @param values the values the matrix is filled with
     */
    public Matrix(double[][] values) {
        this.values = values;
    }

    /*
        Fills the given Matrix with one value in each position
            @param value the value the matrix is filled with
            @return new Matrix filled with value
     */
    public Matrix fill(double value) {
        double[][] result = new double[values.length][values[0].length];
        for (int n = 0; n < values.length; n++)
            for (int m = 0; m < values[0].length; m++)
                result[n][m] = value;
        return new Matrix(result);
    }

    /*
        Copys the given values into a new Matrix
            @param values the values the matrix is filled with
            @return new Matrix with given values
     */
    public Matrix fill(double[][] values) {
        return new Matrix(values);
    }

    /*
        Transposes the given matrix (new format is m x n)
            @return new transposed Matrix
     */
    public Matrix transpose() {
        double[][] transposed = new double[values[0].length][values.length];
        for (int i = 0; i < values.length; i++)
            for (int j = 0; j < values[0].length; j++)
                transposed[j][i] = values[i][j];
        return new Matrix(transposed);
    }

    /*
        Multiplies the matrix elementwise with a scalar
            @param scalar the matrix is multiplied with
            @return new scaled Matrix
     */
    public Matrix multiply(double scalar) {
        double[][] result = new double[values.length][values[0].length];
        for (int i = 0; i < values.length; i++)
            for (int j = 0; j < values[0].length; j++)
                result[i][j] = values[i][j] * scalar;
        return new Matrix(result);
    }

    /*
        Adds one matrix to this matrix elementwise
            @param other the matrix that is added to the current one
            @return new Matrix as the elementwise sum
            @throws IllegalArgumentException if dimensions do not match
     */
    public Matrix add(Matrix other) {
        if (values.length != other.values.length || values[0].length != other.values[0].length)
            throw new IllegalArgumentException(
                "Dimension mismatch: " + values.length + "x" + values[0].length +
                " vs " + other.values.length + "x" + other.values[0].length
            );
        double[][] result = new double[values.length][values[0].length];
        for (int i = 0; i < values.length; i++)
            for (int j = 0; j < values[0].length; j++)
                result[i][j] = values[i][j] + other.values[i][j];
        return new Matrix(result);
    }

    /*
        Adds a vector of length m to each row of the matrix
            @param vector the vector to add to each row
            @return new Matrix with the vector added to each row
            @throws IllegalArgumentException if vector length does not match number of columns
     */
    public Matrix addVector(double[] vector) {
        if (vector.length != values[0].length)
            throw new IllegalArgumentException(
                "Dimension mismatch: vector length " + vector.length +
                " does not match column count " + values[0].length
            );
        double[][] result = new double[values.length][values[0].length];
        for (int i = 0; i < values.length; i++)
            for (int j = 0; j < values[0].length; j++)
                result[i][j] = values[i][j] + vector[j];
        return new Matrix(result);
    }

    /*
        Multiplies two matrices elementwise (Hadamard product)
            @param other the matrix to multiply elementwise with
            @return new Matrix as the elementwise product
            @throws IllegalArgumentException if dimensions do not match
     */
    public Matrix hadamard(Matrix other) {
        if (values.length != other.values.length || values[0].length != other.values[0].length)
            throw new IllegalArgumentException(
                "Dimension mismatch: " + values.length + "x" + values[0].length +
                " vs " + other.values.length + "x" + other.values[0].length
            );
        double[][] result = new double[values.length][values[0].length];
        for (int i = 0; i < values.length; i++)
            for (int j = 0; j < values[0].length; j++)
                result[i][j] = values[i][j] * other.values[i][j];
        return new Matrix(result);
    }

    public double get(int row, int col) {
        return values[row][col];
    }

    public double[][] getValues() {
        return values;
    }

    public int getRows() {
        return values.length;
    }

    public int getColumns() {
        return values[0].length;
    }

    /*
        Subtracts another matrix elementwise
            @param other the matrix to subtract
            @return new Matrix as the elementwise difference
            @throws IllegalArgumentException if dimensions do not match
     */
    public Matrix subtract(Matrix other) {
        if (values.length != other.values.length || values[0].length != other.values[0].length)
            throw new IllegalArgumentException(
                "Dimension mismatch: " + values.length + "x" + values[0].length +
                " vs " + other.values.length + "x" + other.values[0].length
            );
        double[][] result = new double[values.length][values[0].length];
        for (int i = 0; i < values.length; i++)
            for (int j = 0; j < values[0].length; j++)
                result[i][j] = values[i][j] - other.values[i][j];
        return new Matrix(result);
    }

    /*
        Convenience wrapper: standard matrix multiplication A @ B
            @param other the right-hand matrix
            @return new Matrix as the product
     */
    public Matrix matmul(Matrix other) {
        return MatrixMultiplication.multiply(this, other);
    }

    public void getDimensions() {
        System.out.println(values.length + "x" + values[0].length);
    }
}
