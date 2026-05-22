# Power Supply Model

This repository includes a lightweight power-supply response model between the voltage command produced by a policy and the voltage actually applied to the coils.

In the real device, the commanded voltage is not always obtained immediately or exactly. Static gain, offset, communication delay, control latency, and actuator dynamics can create a gap between:

```text
U_set   : voltage command from the controller
U_real  : voltage delivered by the power supply
```

The current implementation exposes this feature through `environment.power_supply.PowerSupplyModel`. `HFMSocketPredictor.step()` passes the 12-dimensional action through the model before sending it to the HFM socket predictor.

## Current Behavior

- The model is channel-wise for 12 coil-voltage channels.
- Each channel applies a static gain `K` and offset `b`.
- Each channel has a small response delay, sampled on reset by default.
- The simulator step is treated as `1 ms`.
- `reset()` clears the delay buffer so each episode starts from a clean power-supply state.

Conceptually:

```text
policy action U_set
  -> power supply response model
  -> actual voltage U_real
  -> HFM predictor
```

This makes the environment slightly closer to the device: policies should not assume that a voltage command is applied perfectly at the same step.

## Example

Use the standalone example to visualize a simple step response:

```bash
python examples/example_power_supply_step.py
```

The generated response figure is:

![Power supply step response](power_supply_step_response.png)

## Notes For Participants

This model is intended as an engineering feature for the competition environment, not as an official scoring formula. Participants can treat it as part of the environment dynamics: the action is still a 12-dimensional voltage command, but the plant receives the delayed/gain-adjusted voltage.
