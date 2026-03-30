import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# import Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import StatevectorSimulator, AerSimulator
from qiskit.visualization import *



def plot_image(Image, title):
    plt.title(title)
    plt.xticks(range(Image.shape[0]))
    plt.yticks(range(Image.shape[1]))
    plt.imshow(Image, extent=[  0,Image.shape[0], Image.shape[1],0,], cmap='grey')
    plt.colorbar()
    plt.show()

def normalize_image(image):
    image64 = np.asarray(image, dtype=np.float64)
    shape = image64.shape
    flat = image64.flatten()
    norm = np.linalg.norm(flat)
    if norm == 0:
        raise ValueError("Input image has zero norm; cannot encode amplitudes.")
    norm_image = flat / norm
    return norm_image.reshape(shape)


class QHED:
    

    def __init__(self, image, measure=False, display=False, plot_edges=False, shots=1024, edge_threshold=1e-5):
        self.image = image
        self.norm_image = normalize_image(image)
        self.norm_image_T = self.norm_image.T
        self.EDGE_THRESHOLD = edge_threshold
        self.measure = measure
        self.display = display
        self.plot_edges = plot_edges
        self.shots = shots

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
        self.hscan = self.pad_statevector(hscan_raw, self.data_dim)
        self.vscan = self.pad_statevector(vscan_raw, self.data_dim)

        # simulator must exist before get_results()
        if self.measure:
            self.simulator = AerSimulator()
        else:
            self.simulator = StatevectorSimulator()

        self.circs = [self.horizontal_scan(), self.vertical_scan()]
        self.results = self.get_results()

    @staticmethod
    def pad_statevector(vector, target_len):
        vector64 = np.asarray(vector, dtype=np.float64)
        if len(vector64) == target_len:
            return vector64
        padded = np.zeros(target_len, dtype=np.float64)
        padded[: len(vector64)] = vector64
        return padded

    @staticmethod
    def prepare_amplitudes(vector):
        """Return a strictly normalized float64 amplitude vector for state prep."""
        amplitudes = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(amplitudes)
        if norm == 0:
            raise ValueError("Amplitude vector has zero norm; cannot initialize circuit state.")
        return amplitudes / norm

    def vertical_scan(self):
        return self._build_scan_circuit(self.vscan)

    def horizontal_scan(self):
        return self._build_scan_circuit(self.hscan)

    def build_scan_circuit(self, scan_vector):
        qc = QuantumCircuit(self.n_qubits)

        # initialize only data qubits (exclude ancilla q0)
        qc.initialize(self._prepare_amplitudes(scan_vector), range(1, self.n_qubits))
        qc.barrier(label="init")
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

    def result_to_flat_probabilities(self, result):
        flat_probs = np.zeros(self.data_dim, dtype=np.float64)

        if self.measure:
            for bitstring, prob in result.items():
                index = int(bitstring, 2)
                if index < self.data_dim:
                    flat_probs[index] = prob
        else:
            probabilities = np.abs(result) ** 2
            flat_probs[: len(probabilities)] = probabilities

        # Drop amplitudes that came only from zero-padding.
        return flat_probs[: self.original_size]

    def result_to_flat_edge(self, result):
        flat_probs = self._result_to_flat_probabilities(result)
        return (flat_probs > self.EDGE_THRESHOLD).astype(int)

    def flat_to_edge_image(self, flat_edge, idx):
        if idx == 0:
            # Horizontal: encoded from image.flatten()
            return flat_edge.reshape(self.rows, self.cols)

        # Vertical: encoded from image.T.flatten(); reshape in transposed frame then map back.
        edge_image_t = flat_edge.reshape(self.cols, self.rows)
        return edge_image_t.T

    def plot_results(self):
        edges = []
        for idx, result in enumerate(self.results):
            flat_edge = self._result_to_flat_edge(result)
            edge_image = self._flat_to_edge_image(flat_edge, idx)

            edges.append(edge_image)
            title = "Horizontal Edges" if idx == 0 else "Vertical Edges"
            #plot individual edge images if desired
            if self.plot_edges:
                plot_image(edge_image, title)

        # combine horizontal and vertical edges
        combined_edges = np.logical_or(edges[0], edges[1]).astype(int)
        plot_image(combined_edges, "Combined Edges")

    def plot_raw_results(self):
        edges = []
        for idx, result in enumerate(self.results):
            flat_edge = self._result_to_flat_probabilities(result)
            edge_image = self._flat_to_edge_image(flat_edge, idx)

            edges.append(edge_image)
            title = "Horizontal Edge Intensity (Raw)" if idx == 0 else "Vertical Edge Intensity (Raw)"
            if self.plot_edges:
                plot_image(edge_image, title)

        combined_edges = np.maximum(edges[0], edges[1])
        plot_image(combined_edges, "Combined Edge Intensity (Raw)")
