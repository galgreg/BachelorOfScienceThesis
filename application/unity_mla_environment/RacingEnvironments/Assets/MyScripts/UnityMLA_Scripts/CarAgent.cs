using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class CarAgent : Agent {
    public void Constructor(
            RaceTrackAcademy aAcademy,
            GameObject aCarObject,
            Vector3 aInitialPosition) {
        mEnvironmentAcademy = aAcademy;
        mCarObject = aCarObject;
        mWheelColliders = CreateWheelColliders(aCarObject);
        mWheelTransforms = CreateWheelTransform(aCarObject);
        agentParameters = new AgentParameters();
        agentParameters.resetOnDone = false;

        mInitialPosition = aInitialPosition;
        mLastPosition = mInitialPosition;
    }
    private List<WheelCollider> CreateWheelColliders(GameObject aCarObject) {
        var wheelColliders = new List<WheelCollider>();
        aCarObject.GetComponentsInChildren<WheelCollider>(wheelColliders);
        wheelColliders.Sort((x, y) => (x.name.CompareTo(y.name)));
        return wheelColliders;
    }
    private List<Transform> CreateWheelTransform(GameObject aCarObject) {
        var wheelTransforms = new List<Transform>(4);
        var carTransform = aCarObject.transform;
        wheelTransforms.Add(carTransform.Find("Wheels/FrontDriver"));
        wheelTransforms.Add(carTransform.Find("Wheels/FrontPassenger"));
        wheelTransforms.Add(carTransform.Find("Wheels/RearDriver"));
        wheelTransforms.Add(carTransform.Find("Wheels/RearPassenger"));
        return wheelTransforms;
    }
    public void SetInputProperties(
            float aMaxSensorLength,
            float aFieldOfView,
            uint aRaysCount) {
        mMaxSensorLength = aMaxSensorLength;
        mFieldOfView = aFieldOfView;
        mRaysCount = aRaysCount;
        mCarInput = new CarAgentInput(
                mCarObject.transform,
                mMaxSensorLength,
                mFieldOfView,
                mRaysCount);
    }
    public void SetOutputProperties(float aMaxSteeringAngle, float aMotorForce) {
        mMaxSteeringAngle = aMaxSteeringAngle;
        mMotorForce = aMotorForce;
        mCarOutput = new CarAgentOutput(mMaxSteeringAngle, mMotorForce);
        mCarOutput.SetWheelColliders(mWheelColliders);
        mCarOutput.SetWheelTransform(mWheelTransforms);
    }
    public override void AgentReset() {
        mEpisodeReward = 0.0f;
        mCarObject.SetActive(true);
    }
    public override void CollectObservations() {
        AddVectorObs(mCarInput.RenderSensorsAndGetNormalizedDistanceList());
        AddVectorObs(GetEpisodeReward());

        mDrivenDistance += Vector3.Distance(transform.position, mLastPosition);
        mLastPosition = transform.position;
    }
    public override void AgentAction(float[] vectorAction, string textAction) {
        mCarOutput.Update(vectorAction[0], vectorAction[1]);
        
    }
    public override void AgentOnDone() {
        // Display episode reward (optional line, only for debug purpose!)
        // Debug.Log("Episode reward: " + GetEpisodeReward());
        mEnvironmentAcademy.IncrementAgentDoneCounter();
        mCarObject.SetActive(false);

        mDrivenDistance = 0.0f;
        mLastPosition = mInitialPosition;
    }

    public void SaveEpisodeReward() {
        mEpisodeReward = mDrivenDistance + GetCumulativeReward();
    }
    public float GetEpisodeReward() {
        return mEpisodeReward;
    }

    private RaceTrackAcademy mEnvironmentAcademy;
    private GameObject mCarObject;
    private List<WheelCollider> mWheelColliders;
    private List<Transform> mWheelTransforms;
    private float mMaxSensorLength = 0.5f;
    private float mFieldOfView = 180.0f;
    private uint mRaysCount = 7;
    private float mMaxSteeringAngle = 30.0f;
    private float mMotorForce = 250.0f;
    private CarAgentInput mCarInput;
    private CarAgentOutput mCarOutput;
    private float mEpisodeReward;
    private Vector3 mLastPosition;
    private float mDrivenDistance = 0.0f;
    private Vector3 mInitialPosition;
}
