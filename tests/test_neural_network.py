import torch
import neural_network
import pytest

def test_save():
    model = neural_network.NeuralNet(1, 1, 1)
    model.save_model(1, 1)
    model.loadModel("model_1_1.pt")
    assert model.input_size == 1
    assert model.hidden_size == 1
    assert model.output_size == 1

def test_overwriting():
    model = neural_network.NeuralNet(1, 1, 1)
    model.save_model(1, 1)
    model.loadModel("model_1_1.pt")
    assert model.input_size == 1
    assert model.hidden_size == 1
    assert model.output_size == 1

    model = neural_network.NeuralNet(2, 2, 2)
    model.save_model(1, 1)
    model.loadModel("model_1_1.pt")
    assert model.input_size == 2
    assert model.hidden_size == 2
    assert model.output_size == 2

def test_load():
    model = neural_network.NeuralNet(1, 1, 1)
    model.save_model(1, 1)
    model.loadModel("model_1_1.pt")
    assert model.input_size == 1
    assert model.hidden_size == 1
    assert model.output_size == 1

