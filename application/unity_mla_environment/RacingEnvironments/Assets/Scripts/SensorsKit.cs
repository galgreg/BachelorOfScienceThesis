using System;
using System.Collections.Generic;
using UnityEngine;

public class SensorsKit : MonoBehaviour {
    // Initialization code
    private void Start() {
        InitSensorList();
        // TODO -> czy trzeba tu coś jeszcze dopisać?
    }
    private void InitSensorList() {
        const uint ANGLE_BETWEEN_SENSORS = FIELD_OF_VIEW / (RAYS_COUNT - 1);
        for (uint i = 0; i < RAYS_COUNT; ++i) {
            uint currentSensorAngle = ANGLE_BETWEEN_SENSORS * i;
            var newSensor = CreateNewSensor(currentSensorAngle);
            mSensorList.Add(newSensor);
        }
    }
    private CarSensor CreateNewSensor(uint aCurrentSensorAngle) {
        var newSensor = new CarSensor();
        Vector3 sensorOrigin = ComputeSensorOrigin();
        Vector3 sensorDirection = ComputeSensorDirection(aCurrentSensorAngle);
        newSensor.SetRayProperties(sensorOrigin, sensorDirection);
        return newSensor;
    }
    private Vector3 ComputeSensorOrigin() {
        throw new NotImplementedException();
    }
    private Vector3 ComputeSensorDirection(uint aCurrentSensorAngle) {
        throw new NotImplementedException();
    }
    private List<double> GetDistanceList() {
        List<double> distanceList = new List<double>();
        foreach (var sensor in mSensorList) {
            distanceList.Add(sensor.GetDistance());
        }
        return distanceList;
    }
    public Transform mCarTransform;
    private const uint FIELD_OF_VIEW = 180;
    private const uint RAYS_COUNT = 3;
    private List<CarSensor> mSensorList;
}
