using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class CarAgent : Agent {
    private void Start() {
        agentParameters.resetOnDone = false;
        mCarInput = new CarAgentInput(
                CarObject.transform,
                MaxSensorLength,
                FieldOfView,
                RaysCount);
        mCarOutput = new CarAgentOutput(MaxSteeringAngle, MotorForce);
        mCarOutput.SetWheelColliders(WheelColliders);
        mCarOutput.SetWheelTransform(WheelTransforms);
    }
    public override void AgentReset() {
        mCarDistance = 0.0f;
        CarObject.SetActive(true);
    }
    public override void CollectObservations() {
        AddVectorObs(mCarInput.RenderSensorsAndGetNormalizedDistanceList());
    }
    public override void AgentAction(float[] vectorAction, string textAction) {
        mCarOutput.Update(vectorAction[0], vectorAction[1]);
    }
    public override void AgentOnDone() {
        EnvironmentAcademy.IncrementAgentDoneCounter();
        CarObject.SetActive(false);
    }

    private void Update() {
        var carRigidbody = CarObject.GetComponent<Rigidbody>();
        mCarDistance += carRigidbody.velocity.magnitude * Time.deltaTime;    
    }

    public float GetFitness() {
        return mCarDistance;
    }

    [Header("Academy Object")]
    public RaceTrackAcademy EnvironmentAcademy;

    [Header("CarAgent Object")]
    public GameObject CarObject;

    [Header("CarAgent Wheels")]
    public List<WheelCollider> WheelColliders;
    public List<Transform> WheelTransforms;
    
    [Header("CarAgent Input")]
    public float MaxSensorLength = 0.5f;
    public float FieldOfView = 180.0f;
    public uint RaysCount = 7;
    
    [Header("CarAgent Output")]
    public float MaxSteeringAngle = 30.0f;
    public float MotorForce = 250.0f;

    
    private float mCarDistance;
    private CarAgentInput mCarInput;
    private CarAgentOutput mCarOutput;
}
