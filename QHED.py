import numpy as np
import matplotlib.pyplot as plt

# import Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import StatevectorSimulator, AerSimulator
from qiskit.visualization import *



def plot_image(Image, title):
    plt.title(title)
    plt.xticks(range(Image.shape[0]))
    plt.yticks(range(Image.shape[1]))
    plt.imshow(Image, extent=[  0,Image.shape[0], Image.shape[1],0,], cmap='hot', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

def normalize_image(image):
    shape = image.shape
    norm_image = image.flatten()/np.linalg.norm(image.flatten())
    return norm_image.reshape(shape)

class QHED:
    def __init__(
        self,
        image,
        measure=False,
        display=False,
        shots=1024,
        use_noise=False,
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        readout_error=0.02,
    ):
        self.image = image
        self.norm_image = normalize_image(image)
        self.norm_image_T = self.norm_image.T

        self.measure = measure
        self.display = display
        self.shots = shots
        self.use_noise = use_noise
        self.single_qubit_error = single_qubit_error
        self.two_qubit_error = two_qubit_error
        self.readout_error = readout_error

        self.rows, self.cols = self.image.shape
        self.original_size = self.rows * self.cols

        # Number of data qubits needed to encode all pixels (padded to power of 2).
        self.data_qubits = int(np.ceil(np.log2(self.original_size)))
        self.data_dim = 2**self.data_qubits

        # +1 ancilla qubit
        self.n_qubits = self.data_qubits + 1
        self.shift_matrix = np.roll(np.identity(2**self.n_qubits), 1, axis=1)

        # Build horizontal/vertical scan vectors and pad to required amplitude length.
        hscan_raw = self.norm_image.flatten()
        vscan_raw = self.norm_image_T.flatten()
        self.hscan = self._pad_statevector(hscan_raw, self.data_dim)
        self.vscan = self._pad_statevector(vscan_raw, self.data_dim)

        # Simulator must exist before get_results().
        # Noise model is supported for shot-based simulation (measure=True).
        if self.use_noise and not self.measure:
            raise ValueError("use_noise=True requires measure=True")

        if self.measure:
            if self.use_noise:
                self.noise_model = self._build_noise_model()
                self.simulator = AerSimulator(noise_model=self.noise_model)
            else:
                self.noise_model = None
                self.simulator = AerSimulator()
        else:
            self.noise_model = None
            self.simulator = StatevectorSimulator()

        self.circs = [self.horizontal_scan(), self.vertical_scan()]
        self.results = self.get_results()

    @staticmethod
    def _pad_statevector(vector, target_len):
        if len(vector) == target_len:
            return vector
        padded = np.zeros(target_len, dtype=vector.dtype)
        padded[: len(vector)] = vector
        return padded

    def _build_noise_model(self):
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

        noise_model = NoiseModel()

        # Gate noise
        err_1q = depolarizing_error(self.single_qubit_error, 1)
        err_2q = depolarizing_error(self.two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(err_1q, ["u", "sx", "x", "h", "rz"])
        noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])

        # Readout noise
        p = self.readout_error
        ro_error = ReadoutError([[1 - p, p], [p, 1 - p]])
        noise_model.add_all_qubit_readout_error(ro_error)

        return noise_model

    def vertical_scan(self):
        qc = QuantumCircuit(self.n_qubits)

        # initialize only data qubits (exclude ancilla q0)
        qc.initialize(self.vscan, range(1, self.n_qubits))
        qc.h(0)
        qc.unitary(self.shift_matrix, range(self.n_qubits), label="Shift")
        qc.h(0)

        if self.measure:
            qc.measure_all()
        if self.display:
            display(qc.draw("mpl"))
        return qc

    def horizontal_scan(self):
        qc = QuantumCircuit(self.n_qubits)

        qc.initialize(self.hscan, range(1, self.n_qubits))
        qc.h(0)
        qc.unitary(self.shift_matrix, range(self.n_qubits), label="Shift")
        qc.h(0)

        if self.measure:
            qc.measure_all()
        if self.display:
            display(qc.draw("mpl"))
        return qc

    def get_results(self):
        results = []
        for qc in self.circs:
            transpiled_qc = transpile(qc, self.simulator)
            result = self.simulator.run(transpiled_qc, shots=self.shots).result()

            if self.measure:
                counts = result.get_counts()
                counts = {k: v / self.shots for k, v in counts.items()}
                # keep only odd states (ancilla qubit q0 = 1 => rightmost bit)
                counts = {k[:-1]: v for k, v in counts.items() if k[-1] == "1"}
                results.append(counts)
            else:
                # keep odd indices (ancilla q0 = 1)
                statevector = result.get_statevector().data[1::2]
                results.append(statevector)

        return results

    def _result_to_flat_edge(self, result):
        threshold = lambda x: 1 if x > 1e-10 else 0
        flat_edge = np.zeros(self.data_dim, dtype=int)

        if self.measure:
            for bitstring, prob in result.items():
                index = int(bitstring, 2)
                if index < self.data_dim:
                    flat_edge[index] = threshold(prob)
        else:
            probabilities = np.abs(result) ** 2
            flat_edge[: len(probabilities)] = (probabilities > 1e-10).astype(int)

        # Drop amplitudes that came only from zero-padding.
        return flat_edge[: self.original_size]

    def plot_results(self):
        edges = []
        for idx, result in enumerate(self.results):
            flat_edge = self._result_to_flat_edge(result)

            if idx == 0:
                # Horizontal: encoded from image.flatten()
                edge_image = flat_edge.reshape(self.rows, self.cols)
            else:
                # Vertical: encoded from image.T.flatten(); reshape in transposed frame then map back.
                edge_image_t = flat_edge.reshape(self.cols, self.rows)
                edge_image = edge_image_t.T

            edges.append(edge_image)

            title = "Horizontal Edges" if idx == 0 else "Vertical Edges"
            plot_image(edge_image, title)

        # combine horizontal and vertical edges
        combined_edges = np.logical_or(edges[0], edges[1]).astype(int)
        plot_image(combined_edges, "Combined Edges")
