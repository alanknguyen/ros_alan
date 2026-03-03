
"use strict";

let EndpointStates = require('./EndpointStates.js');
let EndpointState = require('./EndpointState.js');
let CameraControl = require('./CameraControl.js');
let CameraSettings = require('./CameraSettings.js');
let IOComponentCommand = require('./IOComponentCommand.js');
let CollisionDetectionState = require('./CollisionDetectionState.js');
let EndpointNamesArray = require('./EndpointNamesArray.js');
let NavigatorState = require('./NavigatorState.js');
let IOStatus = require('./IOStatus.js');
let InteractionControlState = require('./InteractionControlState.js');
let IONodeConfiguration = require('./IONodeConfiguration.js');
let URDFConfiguration = require('./URDFConfiguration.js');
let DigitalIOState = require('./DigitalIOState.js');
let JointLimits = require('./JointLimits.js');
let IODataStatus = require('./IODataStatus.js');
let DigitalOutputCommand = require('./DigitalOutputCommand.js');
let AnalogIOStates = require('./AnalogIOStates.js');
let RobotAssemblyState = require('./RobotAssemblyState.js');
let IOComponentStatus = require('./IOComponentStatus.js');
let IONodeStatus = require('./IONodeStatus.js');
let AnalogIOState = require('./AnalogIOState.js');
let NavigatorStates = require('./NavigatorStates.js');
let HomingState = require('./HomingState.js');
let IODeviceStatus = require('./IODeviceStatus.js');
let SEAJointState = require('./SEAJointState.js');
let HomingCommand = require('./HomingCommand.js');
let CollisionAvoidanceState = require('./CollisionAvoidanceState.js');
let IOComponentConfiguration = require('./IOComponentConfiguration.js');
let InteractionControlCommand = require('./InteractionControlCommand.js');
let HeadPanCommand = require('./HeadPanCommand.js');
let JointCommand = require('./JointCommand.js');
let IODeviceConfiguration = require('./IODeviceConfiguration.js');
let AnalogOutputCommand = require('./AnalogOutputCommand.js');
let DigitalIOStates = require('./DigitalIOStates.js');
let HeadState = require('./HeadState.js');
let CalibrationCommandActionGoal = require('./CalibrationCommandActionGoal.js');
let CalibrationCommandFeedback = require('./CalibrationCommandFeedback.js');
let CalibrationCommandGoal = require('./CalibrationCommandGoal.js');
let CalibrationCommandActionResult = require('./CalibrationCommandActionResult.js');
let CalibrationCommandActionFeedback = require('./CalibrationCommandActionFeedback.js');
let CalibrationCommandAction = require('./CalibrationCommandAction.js');
let CalibrationCommandResult = require('./CalibrationCommandResult.js');

module.exports = {
  EndpointStates: EndpointStates,
  EndpointState: EndpointState,
  CameraControl: CameraControl,
  CameraSettings: CameraSettings,
  IOComponentCommand: IOComponentCommand,
  CollisionDetectionState: CollisionDetectionState,
  EndpointNamesArray: EndpointNamesArray,
  NavigatorState: NavigatorState,
  IOStatus: IOStatus,
  InteractionControlState: InteractionControlState,
  IONodeConfiguration: IONodeConfiguration,
  URDFConfiguration: URDFConfiguration,
  DigitalIOState: DigitalIOState,
  JointLimits: JointLimits,
  IODataStatus: IODataStatus,
  DigitalOutputCommand: DigitalOutputCommand,
  AnalogIOStates: AnalogIOStates,
  RobotAssemblyState: RobotAssemblyState,
  IOComponentStatus: IOComponentStatus,
  IONodeStatus: IONodeStatus,
  AnalogIOState: AnalogIOState,
  NavigatorStates: NavigatorStates,
  HomingState: HomingState,
  IODeviceStatus: IODeviceStatus,
  SEAJointState: SEAJointState,
  HomingCommand: HomingCommand,
  CollisionAvoidanceState: CollisionAvoidanceState,
  IOComponentConfiguration: IOComponentConfiguration,
  InteractionControlCommand: InteractionControlCommand,
  HeadPanCommand: HeadPanCommand,
  JointCommand: JointCommand,
  IODeviceConfiguration: IODeviceConfiguration,
  AnalogOutputCommand: AnalogOutputCommand,
  DigitalIOStates: DigitalIOStates,
  HeadState: HeadState,
  CalibrationCommandActionGoal: CalibrationCommandActionGoal,
  CalibrationCommandFeedback: CalibrationCommandFeedback,
  CalibrationCommandGoal: CalibrationCommandGoal,
  CalibrationCommandActionResult: CalibrationCommandActionResult,
  CalibrationCommandActionFeedback: CalibrationCommandActionFeedback,
  CalibrationCommandAction: CalibrationCommandAction,
  CalibrationCommandResult: CalibrationCommandResult,
};
