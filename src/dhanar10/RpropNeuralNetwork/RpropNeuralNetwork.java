package dhanar10.RpropNeuralNetwork;

public class RpropNeuralNetwork {
	public static final int INPUT_NEURON = 2;
	public static final int HIDDEN_NEURON = 4;
	public static final int OUTPUT_NEURON = 1;
	public static final double DELTA_ZERO = 0.1;
	public static final double ETA_PLUS = 1.2;
	public static final double ETA_MINUS = 0.5;
	public static final double DELTA_MAX = 50;
	public static final double DELTA_MIN = 0.000001;
	public static final double TARGET_MSE = 0.001;
	public static final int MAX_EPOCH = 10000;

	public static void main(String[] args) {
		int status = 0;
		
		double yInput[] = new double[INPUT_NEURON];
		double yHidden[] = new double[HIDDEN_NEURON];
		double yOutput[] = new double[OUTPUT_NEURON];
		
		double wInputHidden[][] = new double[INPUT_NEURON][HIDDEN_NEURON];
		double wHiddenOutput[][] = new double[HIDDEN_NEURON][OUTPUT_NEURON];
		
		double dInputHidden[][] = new double[INPUT_NEURON][HIDDEN_NEURON];
		double dHiddenOutput[][] = new double[HIDDEN_NEURON][OUTPUT_NEURON];
		
		double gpInputHidden[][] = new double[INPUT_NEURON][HIDDEN_NEURON];
		double gpHiddenOutput[][] = new double[HIDDEN_NEURON][OUTPUT_NEURON];
		
		int epoch = 0;
		
		double data[][] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}}; // XOR
		
		for (int i = 0; i < wInputHidden.length; i++) {
			for (int j = 0; j < wInputHidden[0].length; j++) {
				wInputHidden[i][j] = Math.random();
			}
		}
		
		for (int i = 0; i < wHiddenOutput.length; i++) {
			for (int j = 0; j < wHiddenOutput[0].length; j++) {
				wHiddenOutput[i][j] = Math.random();
			}
		}
		
		for (int i = 0; i < dInputHidden.length; i++) {
			for (int j = 0; j < dInputHidden[0].length; j++) {
				dInputHidden[i][j] = DELTA_ZERO;
			}
		}
		
		for (int i = 0; i < dHiddenOutput.length; i++) {
			for (int j = 0; j < dHiddenOutput[0].length; j++) {
				dHiddenOutput[i][j] = DELTA_ZERO;
			}
		}
		
		while (true) {
			double mse = 0;
			
			double yTarget[] = new double[OUTPUT_NEURON];
			
			double eHidden[] = new double[HIDDEN_NEURON];
			double eOutput[] = new double[OUTPUT_NEURON];
			
			double gInputHidden[][] = new double[INPUT_NEURON][HIDDEN_NEURON];
			double gHiddenOutput[][] = new double[HIDDEN_NEURON][OUTPUT_NEURON];

			epoch++;
			
			for (double[] d : data) {
				for (int j = 0; j < d.length; j++) {
					if (j < yInput.length) {
						yInput[j] = d[j];
					}
					else {
						yTarget[j - yInput.length] = d[j];
					}
				}
				
				for (int j = 0; j < yHidden.length; j++) {
					yHidden[j] = 0;
					
					for (int k = 0; k < yInput.length; k++) {
						yHidden[j] += yInput[k] * wInputHidden[k][j];
					}
					
					yHidden[j] = sigmoid(yHidden[j]);
				}
				
				yOutput[0] = 0;
				
				for (int j = 0; j < yHidden.length; j++) {
					yOutput[0] += yHidden[j] * wHiddenOutput[j][0];
				}
				
				yOutput[0] = sigmoid(yOutput[0]);
				
				eOutput[0] = (yTarget[0] - yOutput[0]) * dsigmoid(yOutput[0]);
				
				for (int j = 0; j < yHidden.length; j++) {
					eHidden[j] = eOutput[0] * wHiddenOutput[j][0] *  dsigmoid(yHidden[j]);
				}
				
				for (int j = 0; j < yHidden.length; j++) {
					for (int k = 0; k < yInput.length; k++) {
						gInputHidden[k][j] += eHidden[j] * yInput[k];
					}
				}
				
				for (int j = 0; j < yHidden.length; j++) {
					gHiddenOutput[j][0] += eOutput[0] * yHidden[j];
				}
				
				mse += Math.pow(yTarget[0] - yOutput[0], 2);
			}
			
			for (int j = 0; j < yHidden.length; j++) {
				for (int k = 0; k < yInput.length; k++) {
					double change = gpInputHidden[k][j] * gInputHidden[k][j];
					
					if (change > 0) {
						dInputHidden[k][j] = Math.min(dInputHidden[k][j] * ETA_PLUS, DELTA_MAX);
						wInputHidden[k][j] += Math.signum(gInputHidden[k][j])  * dInputHidden[k][j];
						gpInputHidden[k][j] = gInputHidden[k][j];
					}
					else if (change < 0) {
						dInputHidden[k][j] = Math.max(dInputHidden[k][j] * ETA_MINUS, DELTA_MIN);
						gpInputHidden[k][j] = 0;
					}
					else if (change == 0) {
						wInputHidden[k][j] += Math.signum(gInputHidden[k][j]) * dInputHidden[k][j];
						gpInputHidden[k][j] = gInputHidden[k][j];
					}
				}
			}
			
			for (int j = 0; j < yHidden.length; j++) {
				double change = gpHiddenOutput[j][0] * gHiddenOutput[j][0];
				
				if (change > 0) {
					dHiddenOutput[j][0] = Math.min(dHiddenOutput[j][0] * ETA_PLUS, DELTA_MAX);
					wHiddenOutput[j][0] += Math.signum(gHiddenOutput[j][0]) * dHiddenOutput[j][0];
					gpHiddenOutput[j][0] = gHiddenOutput[j][0];
				}
				else if (change < 0) {
					dHiddenOutput[j][0] = Math.max(dHiddenOutput[j][0] * ETA_MINUS, DELTA_MIN);
					gpHiddenOutput[j][0] = 0;
				}
				else if (change == 0) {
					wHiddenOutput[j][0] += Math.signum(gHiddenOutput[j][0]) * dHiddenOutput[j][0];
					gpHiddenOutput[j][0] = gHiddenOutput[j][0];
				}
			}
			
			mse /= data.length;
			
			System.out.println(epoch + "\t" + mse);
			
			if (mse < TARGET_MSE) {
				break;
			}
			
			if (epoch == MAX_EPOCH) {
				status = 1;
				break;
			}
		}
		
		System.out.println();
		
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < yInput.length; j++) {
				yInput[j] = data[i][j];
			}
			
			for (int j = 0; j < yHidden.length; j++) {
				yHidden[j] = 0;
				
				for (int k = 0; k < yInput.length; k++) {
					yHidden[j] += yInput[k] * wInputHidden[k][j];
				}
				
				yHidden[j] = sigmoid(yHidden[j]);
			}
			
			yOutput[0] = 0;
			
			for (int j = 0; j < yHidden.length; j++) {
				yOutput[0] += yHidden[j] * wHiddenOutput[j][0];
			}
			
			yOutput[0] = sigmoid(yOutput[0]);
			
			System.out.println(yInput[0] + "\t" + yInput[1] + "\t" + yOutput[0]);
		}
		
		System.exit(status);
	}
	
	private static double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}
	
	private static double dsigmoid(double x) {
		return x * (1 - x);
	}
}
