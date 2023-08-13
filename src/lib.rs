mod train;
extern crate rand;


pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
    output: f64,
    input_sum: f64,
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

pub struct Network {
    layers: Vec<Layer>,
}

#[derive(Clone)]
pub struct Gradient {
    weight: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn random(input_size: usize) -> Self {
        let weights = (0..input_size).map(|_| rand::random::<f64>()).collect();
        let bias = rand::random::<f64>();
        Neuron { weights, bias, output: 0.0, input_sum: 0.0 }
    }
}

impl Network {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let hidden_layer = Layer {
            neurons: (0..hidden_size).map(|_| Neuron::random(input_size)).collect(),
        };

        let output_layer = Layer {
            neurons: (0..output_size).map(|_| Neuron::random(hidden_size)).collect(),
        };

        Network {
            layers: vec![hidden_layer, output_layer],
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut layer_input = inputs;

        for layer in &mut self.layers {
            let mut layer_output = vec![];

            for neuron in &mut layer.neurons {
                let sum: f64 = neuron.weights.iter().zip(layer_input.iter()).map(|(a, b)| a * b).sum();
                neuron.input_sum = sum;
                let output = relu(neuron.input_sum + neuron.bias);
                neuron.output = output;
                layer_output.push(output);
            }

            layer_input = layer_output;  // output of this layer is input to the next layer
        }

        // Note: we're now returning the final layer's outputs directly from the neurons, not via the intermediate layer_input vector
        self.layers.last().unwrap().neurons.iter().map(|neuron| neuron.output).collect()
    }

    pub fn backpropagate(&mut self, expected_output: Vec<f64>) {
        let mut errors = self.layers.last().unwrap().neurons.iter().zip(expected_output.iter())
            .map(|(n, &e)| e - n.output)
            .collect::<Vec<_>>(); 

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let mut gradients = Vec::with_capacity(layer.neurons.len());
            let mut new_errors = vec![0.0; layer.neurons[0].weights.len()];

            for (j, neuron) in layer.neurons.iter_mut().enumerate() {
                let error = errors[j];
                gradients.push(error * relu_derivative(neuron.input_sum));

                for (k, weight) in neuron.weights.iter_mut().enumerate() {
                    new_errors[k] += error * *weight;
                    *weight += neuron.output * gradients.last().unwrap();
                }

                neuron.bias += gradients.last().unwrap();
            }

            errors = new_errors; 
        }
    }

}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

//testing
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut network = Network::new(3, 3, 1);

        let inputs = vec![0.5, 0.1, 0.8];
        let expected_output = vec![0.7];

        network.forward(inputs.clone());
        network.backpropagate(expected_output.clone());
        
        let final_output = network.forward(inputs);
        assert!((final_output[0]-expected_output[0]).abs() < 1.0, "Training decreases error");
    }
}