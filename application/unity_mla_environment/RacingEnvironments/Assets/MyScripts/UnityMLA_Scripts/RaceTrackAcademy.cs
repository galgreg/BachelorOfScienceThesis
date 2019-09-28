using System;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class RaceTrackAcademy : Academy {
    public override void InitializeAcademy() {
        mDoneAgentsCounter = 0;
        mStartAgentRotationVector = new Vector3(0.0f, StartAgentRotation, 0.0f);
        mAgentScaleVector =
                new Vector3(CarAgentScale, CarAgentScale, CarAgentScale);
        mAgentList = CreateNewAgentList();
    }
    private List<GameObject> CreateNewAgentList() {
        var newAgentList = new List<GameObject>(PopulationSize);
        for (int i = 0; i < PopulationSize; ++i) {
            var newAgent = CreateNewAgent();
            newAgentList.Add(newAgent);
        }
        return newAgentList;
    }
    private GameObject CreateNewAgent() {
        var newAgent = Instantiate(CarAgentPrefab);
        SetAgentTransform(newAgent);
        PrepareCarAgentComponent(newAgent);
        return newAgent;
    }
    private void SetAgentTransform(GameObject aAgentObject) {
        aAgentObject.transform.position = StartAgentPosition;
        aAgentObject.transform.eulerAngles = mStartAgentRotationVector;
        aAgentObject.transform.localScale = mAgentScaleVector;
    }
    private void PrepareCarAgentComponent(GameObject aAgentObject) {
        var carAgentComponent = aAgentObject.AddComponent<CarAgent>();
        carAgentComponent.Constructor(this, aAgentObject, RewardPerStep);
        carAgentComponent.SetInputProperties(MaxSensorLength, FieldOfView, RaysCount);
        carAgentComponent.SetOutputProperties(MaxSteeringAngle, MotorForce);
        carAgentComponent.AgentReset();
        carAgentComponent.GiveBrain(CarAgentBrain);
    }
    
    public override void AcademyReset() {
        mDoneAgentsCounter = 0;
        for(int i = 0; i < PopulationSize; ++i) {
            SetAgentTransform(mAgentList[i]);
            var carAgentComponent = mAgentList[i].GetComponent<CarAgent>();
            carAgentComponent.AgentReset();
        }
    }

    public override void AcademyStep() {
        if (mDoneAgentsCounter >= PopulationSize) {
            Done();
        }
    }
    public void IncrementAgentDoneCounter() {
        ++mDoneAgentsCounter;
    }

    [Header("Brains")]
    public Brain CarAgentBrain;
    
    [Header("CarAgent Object")]
    public GameObject CarAgentPrefab;

    [Header("CarAgent transform")]
    public Vector3 StartAgentPosition = new Vector3(0.0f, 0.0f, 0.0f);
    public float StartAgentRotation = 0.0f;
    public float CarAgentScale = 0.2f;
    
    [Header("CarAgent Input")]
    public float MaxSensorLength = 0.5f;
    public float FieldOfView = 180.0f;
    public uint RaysCount = 7;
    
    [Header("CarAgent Output")]
    public float MaxSteeringAngle = 30.0f;
    public float MotorForce = 250.0f;

    [Header("Learning parameters")]
    public float RewardPerStep = -0.001f;
    public int PopulationSize = 100;
    
    private List<GameObject> mAgentList;
    private uint mDoneAgentsCounter;

    private Vector3 mStartAgentRotationVector;
    private Vector3 mAgentScaleVector;
}