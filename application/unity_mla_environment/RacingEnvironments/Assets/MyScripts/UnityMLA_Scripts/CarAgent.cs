﻿using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class CarAgent : Agent {
    public void Constructor(
            RaceTrackAcademy aAcademy,
            GameObject aCarObject,
            float aRewardPerStep) {
        mEnvironmentAcademy = aAcademy;
        mCarObject = aCarObject;
        mRewardPerStep = aRewardPerStep;
        mWheelColliders = CreateWheelColliders(aCarObject);
        mWheelTransforms = CreateWheelTransform(aCarObject);
        agentParameters.resetOnDone = false;
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
        wheelTransforms[0] = carTransform.Find("Wheels/FrontDriver");
        wheelTransforms[1] = carTransform.Find("Wheels/FrontPassenger");
        wheelTransforms[2] = carTransform.Find("Wheels/RearDriver");
        wheelTransforms[3] = carTransform.Find("Wheels/RearPassenger");
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
    }
    public override void AgentAction(float[] vectorAction, string textAction) {
        mCarOutput.Update(vectorAction[0], vectorAction[1]);
        AddReward(mRewardPerStep);
    }
    public override void AgentOnDone() {
        mEnvironmentAcademy.IncrementAgentDoneCounter();
        mCarObject.SetActive(false);
    }

    public void SaveEpisodeReward() {
        mEpisodeReward = GetCumulativeReward();
    }
    public float GetEpisodeReward() {
        return mEpisodeReward;
    }

    private RaceTrackAcademy mEnvironmentAcademy;
    private GameObject mCarObject;
    private float mRewardPerStep;
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
}
