# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: shared_types.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12shared_types.proto\"\xe4\x01\n\x05Query\x12\x1d\n\x05\x61udio\x18\x01 \x01(\x0b\x32\x0c.Query.AudioH\x00\x12\x1d\n\x05image\x18\x02 \x01(\x0b\x32\x0c.Query.ImageH\x00\x12\x1d\n\x05video\x18\x03 \x01(\x0b\x32\x0c.Query.VideoH\x00\x12\x1b\n\x04text\x18\x04 \x01(\x0b\x32\x0b.Query.TextH\x00\x1a\x16\n\x05\x41udio\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x1a\x16\n\x05Image\x12\r\n\x05image\x18\x01 \x01(\x0c\x1a\x16\n\x05Video\x12\r\n\x05video\x18\x01 \x01(\x0c\x1a\x14\n\x04Text\x12\x0c\n\x04text\x18\x01 \x01(\tB\x03\n\x01q\"\x93\x05\n\x06Result\x12\x1e\n\x05label\x18\x01 \x01(\x0b\x32\r.Result.LabelH\x00\x12*\n\x0b\x62oundingbox\x18\x02 \x01(\x0b\x32\x13.Result.BoundingBoxH\x00\x12\r\n\x05other\x18\x03 \x01(\t\x1a%\n\x05Label\x12\r\n\x05label\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x1aH\n\x0e\x42oundingBoxBox\x12\x0c\n\x04xmin\x18\x01 \x01(\x02\x12\x0c\n\x04xmax\x18\x02 \x01(\x02\x12\x0c\n\x04ymin\x18\x03 \x01(\x02\x12\x0c\n\x04ymax\x18\x04 \x01(\x02\x1aP\n\x0b\x42oundingBox\x12#\n\x03\x62ox\x18\x01 \x01(\x0b\x32\x16.Result.BoundingBoxBox\x12\r\n\x05label\x18\x02 \x01(\t\x12\r\n\x05score\x18\x03 \x01(\x02\x1a:\n\x16SpeechRecognitionChunk\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x12\n\ntimestamps\x18\x02 \x03(\x02\x1a!\n\x11SpeechRecognition\x12\x0c\n\x04text\x18\x01 \x01(\t\x1a$\n\x04Mask\x12\r\n\x05score\x18\x01 \x01(\x02\x12\r\n\x05label\x18\x02 \x01(\t\x1a)\n\tTokenMask\x12\r\n\x05score\x18\x01 \x01(\x02\x12\r\n\x05token\x18\x02 \x01(\t\x1aR\n\x08TokenNER\x12\r\n\x05group\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x12\x0c\n\x04word\x18\x03 \x01(\t\x12\r\n\x05start\x18\x04 \x01(\x05\x12\x0b\n\x03\x65nd\x18\x05 \x01(\x05\x1a\x43\n\x06\x41nswer\x12\r\n\x05score\x18\x01 \x01(\x02\x12\r\n\x05start\x18\x02 \x01(\x05\x12\x0b\n\x03\x65nd\x18\x03 \x01(\x05\x12\x0e\n\x06\x61nswer\x18\x04 \x01(\t\x1a\x1d\n\x0bTableAnswer\x12\x0e\n\x06\x61nswer\x18\x01 \x01(\tB\x03\n\x01r\"X\n\x05Input\x12!\n\tdata_type\x18\x01 \x01(\x0e\x32\x0e.InputDataType\x12,\n\x0erepresentation\x18\x02 \x01(\x0e\x32\x14.InputRepresentation\"[\n\x06Output\x12\"\n\tdata_type\x18\x01 \x01(\x0e\x32\x0f.OutputDataType\x12-\n\x0erepresentation\x18\x02 \x01(\x0e\x32\x15.OutputRepresentation\"&\n\x04Task\x12\n\n\x02id\x18\x01 \x01(\t\x12\x12\n\nmodel_name\x18\x03 \x01(\t*D\n\rInputDataType\x12\t\n\x05IMAGE\x10\x00\x12\t\n\x05VIDEO\x10\x01\x12\t\n\x05\x41UDIO\x10\x02\x12\x08\n\x04TEXT\x10\x03\x12\x08\n\x04TEST\x10\x04*\x1e\n\x13InputRepresentation\x12\x07\n\x03\x43\x41R\x10\x00*\x1d\n\x0bInputDomain\x12\x0e\n\nSCIENTIFIC\x10\x00*,\n\x0eOutputDataType\x12\x0f\n\x0b\x42OUNDINGBOX\x10\x00\x12\t\n\x05LABEL\x10\x01* \n\x14OutputRepresentation\x12\x08\n\x04\x46\x41\x43\x45\x10\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'shared_types_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _INPUTDATATYPE._serialized_start=1138
  _INPUTDATATYPE._serialized_end=1206
  _INPUTREPRESENTATION._serialized_start=1208
  _INPUTREPRESENTATION._serialized_end=1238
  _INPUTDOMAIN._serialized_start=1240
  _INPUTDOMAIN._serialized_end=1269
  _OUTPUTDATATYPE._serialized_start=1271
  _OUTPUTDATATYPE._serialized_end=1315
  _OUTPUTREPRESENTATION._serialized_start=1317
  _OUTPUTREPRESENTATION._serialized_end=1349
  _QUERY._serialized_start=23
  _QUERY._serialized_end=251
  _QUERY_AUDIO._serialized_start=154
  _QUERY_AUDIO._serialized_end=176
  _QUERY_IMAGE._serialized_start=178
  _QUERY_IMAGE._serialized_end=200
  _QUERY_VIDEO._serialized_start=202
  _QUERY_VIDEO._serialized_end=224
  _QUERY_TEXT._serialized_start=226
  _QUERY_TEXT._serialized_end=246
  _RESULT._serialized_start=254
  _RESULT._serialized_end=913
  _RESULT_LABEL._serialized_start=355
  _RESULT_LABEL._serialized_end=392
  _RESULT_BOUNDINGBOXBOX._serialized_start=394
  _RESULT_BOUNDINGBOXBOX._serialized_end=466
  _RESULT_BOUNDINGBOX._serialized_start=468
  _RESULT_BOUNDINGBOX._serialized_end=548
  _RESULT_SPEECHRECOGNITIONCHUNK._serialized_start=550
  _RESULT_SPEECHRECOGNITIONCHUNK._serialized_end=608
  _RESULT_SPEECHRECOGNITION._serialized_start=610
  _RESULT_SPEECHRECOGNITION._serialized_end=643
  _RESULT_MASK._serialized_start=645
  _RESULT_MASK._serialized_end=681
  _RESULT_TOKENMASK._serialized_start=683
  _RESULT_TOKENMASK._serialized_end=724
  _RESULT_TOKENNER._serialized_start=726
  _RESULT_TOKENNER._serialized_end=808
  _RESULT_ANSWER._serialized_start=810
  _RESULT_ANSWER._serialized_end=877
  _RESULT_TABLEANSWER._serialized_start=879
  _RESULT_TABLEANSWER._serialized_end=908
  _INPUT._serialized_start=915
  _INPUT._serialized_end=1003
  _OUTPUT._serialized_start=1005
  _OUTPUT._serialized_end=1096
  _TASK._serialized_start=1098
  _TASK._serialized_end=1136
# @@protoc_insertion_point(module_scope)