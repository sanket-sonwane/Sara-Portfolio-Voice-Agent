/**
 * Minimal PCM tap for the input graph.
 *
 * It copies each render quantum and posts it to the main thread. The main
 * thread owns resampling and framing, so this processor stays allocation-light
 * and has no knowledge of the server protocol.
 */
class SarangPcmProcessor extends AudioWorkletProcessor {
  process(inputs: Float32Array[][]): boolean {
    const channel = inputs[0]?.[0];
    if (channel && channel.length) {
      this.port.postMessage(channel.slice(0));
    }
    return true;
  }
}

registerProcessor("sarang-pcm", SarangPcmProcessor);
