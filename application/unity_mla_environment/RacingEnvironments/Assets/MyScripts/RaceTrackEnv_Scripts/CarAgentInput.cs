using System.Collections.Generic;
using UnityEngine;

public class CarAgentInput {
    public CarAgentInput(
            Transform aCarTransform,
            float aMaxSensorLength,
            float aFieldOfView,
            uint aRaysCount) {
        mCarTransform = aCarTransform;
        MAX_SENSOR_LENGTH = aMaxSensorLength;
        if (aFieldOfView > MAX_POSSIBLE_FOV) {
            CURRENT_FOV = MAX_POSSIBLE_FOV;
        } else {
            CURRENT_FOV = aFieldOfView;
        }
        RAYS_COUNT = aRaysCount;
        ANGLE_BETWEEN_SENSORS = CURRENT_FOV / (RAYS_COUNT - 1);
        STARTING_ANGLE = (MAX_POSSIBLE_FOV - CURRENT_FOV) / 2;
        mTransformComputer =
                new SensorPropertiesComputer(
                        mCarTransform,
                        MAX_SENSOR_LENGTH * UNITY_ANTIBUG_FACTOR,
                        SENSORS_OFFSET_Y,
                        SENSORS_OFFSET_Z);
        InitSensorList();
    }
    
    public List<float> RenderSensorsAndGetNormalizedDistanceList() {
        for (int i = 0; i < RAYS_COUNT; ++i) {
            float currentSensorAngle = ANGLE_BETWEEN_SENSORS * i + STARTING_ANGLE;
            Vector3 sensorOrigin = mTransformComputer.ComputeSensorOrigin();
            Vector3 sensorDirection =
                    mTransformComputer.ComputeSensorDirection(currentSensorAngle);
            mSensorList[i].SetRayProperties(sensorOrigin, sensorDirection);
            mSensorList[i].Render();
        }
        return GetNormalizedDistanceList();
    }
    
    private void InitSensorList() {
        mSensorList = new List<CarSensor>();
        for (uint i = 0; i < RAYS_COUNT; ++i) {
            float currentSensorAngle = ANGLE_BETWEEN_SENSORS * i;
            var newSensor = CreateNewSensor(currentSensorAngle);
            newSensor.SetSensorParent(mCarTransform);
            mSensorList.Add(newSensor);
        }
    }
    private CarSensor CreateNewSensor(float aCurrentSensorAngle) {
        var newSensor = new CarSensor(
                MAX_SENSOR_LENGTH * UNITY_ANTIBUG_FACTOR,
                UNITY_ANTIBUG_FACTOR);
        Vector3 sensorOrigin = mTransformComputer.ComputeSensorOrigin();
        Vector3 sensorDirection =
                mTransformComputer.ComputeSensorDirection(aCurrentSensorAngle);
        newSensor.SetRayProperties(sensorOrigin, sensorDirection);
        return newSensor;
    }
    private List<float> GetNormalizedDistanceList() {
        List<float> distanceList = new List<float>();
        foreach (var sensor in mSensorList) {
            float tempDistance = sensor.GetNormalizedDistance();
            distanceList.Add(tempDistance);
        }
        return distanceList;
    }

    private readonly Transform mCarTransform;
    private readonly float MAX_SENSOR_LENGTH;
    private readonly float CURRENT_FOV;
    private readonly uint RAYS_COUNT;
    
    private const float SENSORS_OFFSET_Y = 0.03f;
    private const float SENSORS_OFFSET_Z = 0.13f;
    private const uint UNITY_ANTIBUG_FACTOR = 500;
    private const float MAX_POSSIBLE_FOV = 180.0f;
    private readonly float ANGLE_BETWEEN_SENSORS;
    private readonly float STARTING_ANGLE;
    
    private List<CarSensor> mSensorList;
    private SensorPropertiesComputer mTransformComputer;
}
