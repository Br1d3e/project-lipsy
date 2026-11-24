"""Run with python3 bot-1.8.9/tests/runtime_trace_test.py."""
import ast
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
JAVA = ROOT / 'bot-1.8.9/src/main/java/com/bot/projectlipsy'


class RuntimeTraceTest(unittest.TestCase):
    def test_raw_field_names_match_socket_protocol(self):
        server = ast.parse((ROOT / 'models/socket_server.py').read_text())
        names = next(ast.literal_eval(node.value) for node in server.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'OBS_FEATURES'
                             for t in node.targets))
        source = (JAVA / 'ModelConnect.java').read_text()
        constants = dict((name, int(index)) for name, index in re.findall(
            r'private static final int (\w+) = (\d+);', source))
        fields = re.findall(r'frame.field\("(\w+)", observationAvailable \? batch\[(\w+)\]', source)
        self.assertEqual([name for name, _ in fields], names)
        for index, (name, constant) in enumerate(fields):
            self.assertEqual(name, constant)
            self.assertEqual(constants[constant], index)

    def test_json_and_writer_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = root / 'TraceTest.java'
            harness.write_text('''
import java.io.File;
import com.bot.projectlipsy.recordWriter.*;
public class TraceTest {
    public static void main(String[] args) throws Exception {
        RuntimeFrameWriter writer = new RuntimeFrameWriter();
        writer.start(new File(args[0]));
        File first = writer.getFile();
        writer.start(new File(args[0]));
        if (!first.equals(writer.getFile())) throw new AssertionError("reopened active trace");
        float[] actions = {0, 0, 0, 0, 0, 0, 2, .1f, .2f, .6f, .03f, .04f, .03f};
        JsonLineBuilder frame = new JsonLineBuilder(256).field("tick", 1);
        RuntimeFrameWriter.appendModelOutput(frame, actions);
        frame.field("myHp", Float.NaN).field("overrideRule", "quoted\\\"\\nrule");
        writer.writeFrame(frame);
        writer.close();
        writer.close();
        if (writer.isRecording()) throw new AssertionError("still recording");
        writer.start(new File(args[0]));
        if (first.equals(writer.getFile())) throw new AssertionError("overwrote old trace");
        frame = new JsonLineBuilder(256).field("tick", 2);
        RuntimeFrameWriter.appendModelOutput(frame, null);
        writer.writeFrame(frame);
        // The shutdown hook must flush this last frame without an explicit close.
    }
}
''')
            subprocess.run(['javac', '-d', directory,
                            str(JAVA / 'recordWriter/JsonLineBuilder.java'),
                            str(JAVA / 'recordWriter/RuntimeFrameWriter.java'), str(harness)], check=True)
            subprocess.run(['java', '-cp', directory, 'TraceTest', directory], check=True)
            paths = list(root.glob('runtime-traces/*/frames.jsonl'))
            self.assertEqual(len(paths), 2)
            rows = sorted([json.loads(p.read_text()) for p in paths], key=lambda r: r['tick'])
            self.assertEqual(rows[0]['modelState'], 'pearl')
            self.assertAlmostEqual(rows[0]['modelStateProbabilities']['pearl'], .6)
            self.assertIsNone(rows[0]['myHp'])
            self.assertEqual(rows[0]['overrideRule'], 'quoted"\nrule')
            self.assertIsNone(rows[1]['modelState'])
            self.assertTrue(all(v is None for v in rows[1]['modelStateProbabilities'].values()))


if __name__ == '__main__':
    unittest.main()
